"""Phase A2: improved AR supervised training.

Fixes vs the original Stage 2:

1. **Data source.** Original used AV-generated drawings from *arbitrary text snippets*
   (e.g. "The capital of France is"). Those drawings were random scribbles because
   Stage-1 AV's prior is to depict concepts (cat/dog/...), and "capital of France"
   isn't a concept-it-was-trained-on. So AR learned a bad mapping (random scribble →
   random activation).

   v2 uses **real QuickDraw drawings paired with the activation of their caption**.
   For each (caption, real_stroke_sequence) in the SFT corpus:
     - render(real_stroke_sequence) → 224×224 PNG of a recognisable concept
     - h_target = Gemma 4 forward(caption) at layer ℓ, final-token position
     - train AR: PNG → h_target

   AR now learns "image of a cat → activation of 'a drawing of a cat'".

2. **Much more data.** Original was 200 steps × batch 2 = ~400 pairs. v2 is
   2000-5000 steps × batch 8 = 16k-40k pairs. AR's Linear(d,d) actually trains.

3. **Optional: 2-layer MLP head** instead of Linear(d,d) for more capacity.

4. **Pre-cache the rendered images** so step time is dominated by AR forward,
   not by rendering (which we do once per drawing rather than per epoch).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.reconstructor import TruncatedGemmaAR  # noqa: E402
from render import render as stroke_render  # noqa: E402
from stroke_tokenizer import PEN_DOWN, PEN_END, PEN_UP, Stroke  # noqa: E402


def load_sft_corpus(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def strokes_from_dicts(stroke_dicts: list[dict]) -> list[Stroke]:
    return [Stroke(dx=s["dx"], dy=s["dy"], pen=s["pen"]) for s in stroke_dicts]


class MLP2Head(nn.Module):
    """Optional 2-layer MLP head as alternative to Linear(d, d)."""
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d * 2, d)
        # init close to a useful starting point
        with torch.no_grad():
            self.fc1.weight.normal_(mean=0.0, std=0.02)
            self.fc1.bias.zero_()
            self.fc2.weight.normal_(mean=0.0, std=0.02)
            self.fc2.bias.zero_()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


@torch.no_grad()
def extract_caption_activations(model, tokenizer, captions: list[str], layer: int,
                                 device: str = "cuda", batch_size: int = 32) -> torch.Tensor:
    """Run Gemma 4 forward on each caption, extract h_ℓ at final-token position.
    Returns shape (N, hidden)."""
    n = len(captions)
    rows = []
    for start in range(0, n, batch_size):
        batch = captions[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer]  # (B, T, hidden)
        attn = enc["attention_mask"]
        final_pos = attn.sum(dim=1) - 1
        gathered = h[torch.arange(h.shape[0]), final_pos]
        rows.append(gathered.float().cpu())
        if (start // batch_size) % 20 == 0:
            print(f"  [extract] {start}/{n}", flush=True)
    return torch.cat(rows, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sft-corpus", type=Path, default=Path("data/sft_quickdraw.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/ar_v2"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--n-train", type=int, default=8000, help="how many SFT corpus rows to use")
    parser.add_argument("--canvas-size", type=int, default=224)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"L{args.layer:02d}_train.jsonl"

    # Load SFT corpus
    print(f"[ar2] loading SFT corpus {args.sft_corpus}", flush=True)
    rows = load_sft_corpus(args.sft_corpus)[: args.n_train]
    print(f"[ar2] {len(rows)} rows kept", flush=True)

    # Build AR
    print(f"[ar2] building AR at layer {args.layer}", flush=True)
    ar = TruncatedGemmaAR.from_pretrained(args.model_id, layer_ell=args.layer, device="cuda")
    for p in ar.backbone.parameters():
        p.requires_grad = False
    hidden = ar.hidden_size
    if args.head_type == "linear":
        head = ar.linear  # already a Linear(d, d) inside ar
        for p in head.parameters():
            p.requires_grad = True
        head_params = list(head.parameters())
    else:
        # Replace ar.linear with an MLP2Head and re-route the forward to use it
        ar.linear = MLP2Head(hidden).cuda().to(next(ar.backbone.parameters()).dtype)
        head_params = list(ar.linear.parameters())

    print(f"[ar2] head type={args.head_type}, trainable params={sum(p.numel() for p in head_params)/1e6:.2f}M", flush=True)
    optim = torch.optim.AdamW(head_params, lr=args.lr)

    # Extract activations for all captions (one-shot)
    captions = [r["caption"] for r in rows]
    print(f"[ar2] extracting h_ℓ for {len(captions)} captions", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    target = ar.backbone.cuda()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    h_all = extract_caption_activations(target, tok, captions, args.layer)
    print(f"[ar2] activations shape={tuple(h_all.shape)} mean_norm={float(h_all.norm(dim=1).mean()):.2f}", flush=True)

    # Pre-render images (one-shot)
    print(f"[ar2] pre-rendering {len(rows)} drawings", flush=True)
    images = []
    for i, r in enumerate(rows):
        strokes = strokes_from_dicts(r["strokes"])
        img = stroke_render(strokes, canvas_size=args.canvas_size)
        images.append(img)
        if i % 1000 == 0:
            print(f"  [render] {i}/{len(rows)}", flush=True)

    n = len(rows)
    perm = torch.randperm(n).tolist()
    cursor = 0
    t_start = time.time()

    with open(log_path, "a") as log_f:
        for step in range(args.steps):
            if cursor + args.batch_size > n:
                perm = torch.randperm(n).tolist()
                cursor = 0
            batch_idx = perm[cursor : cursor + args.batch_size]
            cursor += args.batch_size

            batch_imgs = [images[i] for i in batch_idx]
            h_target = h_all[batch_idx].cuda()
            h_hat = ar.forward(batch_imgs)
            loss = F.mse_loss(h_hat.float(), h_target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                msg = {"step": step, "loss": float(loss.item()), "elapsed_sec": round(time.time() - t_start, 1)}
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()
                print(f"[ar2] step {step:5d} loss={msg['loss']:.4f}  ({msg['elapsed_sec']:.0f}s)", flush=True)

            if step % args.save_every == 0 and step > 0:
                save_dir = args.out_dir / f"L{args.layer:02d}" / f"step_{step:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"linear": ar.linear.state_dict(), "head_type": args.head_type},
                           save_dir / "head.pt")
                print(f"[ar2] saved → {save_dir}", flush=True)

    final = args.out_dir / f"L{args.layer:02d}" / "final"
    final.mkdir(parents=True, exist_ok=True)
    torch.save({"linear": ar.linear.state_dict(), "head_type": args.head_type}, final / "head.pt")
    print(f"[ar2] DONE → {final}", flush=True)


if __name__ == "__main__":
    main()
