"""Stage 2 — Activation Reconstructor supervised training.

Given:
    * A Stage-1-trained AV (frozen for this stage)
    * An activation corpus: (text → h_ℓ) pairs at the target anchor layer

We train the AR to reconstruct h_ℓ from a rendered drawing of the text:

    1. For each text in the activation corpus, summarize it to a short caption
       (or use the text directly if it's already short).
    2. Run AV (frozen) to generate a drawing from that caption (NO activation
       injection at this stage — we want a "concept" drawing).
    3. Render the drawing to a 224×224 PNG.
    4. Pass the PNG through AR; compute MSE(h, ĥ).
    5. Backprop into AR's LoRA + Linear(d,d) head only.

This calibrates the AR to the AV's drawing distribution, so that when Stage 3
RL turns on activation injection, the AR is already in the right ballpark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ar.reconstructor import TruncatedGemmaAR  # noqa: E402
from av.stroke_decoder import StrokeDecoder  # noqa: E402
from render import render as stroke_render  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--av-ckpt", type=Path, required=True,
                        help="Path to Stage-1 AV checkpoint dir (with adapter_config.json + stroke_vocab.pt)")
    parser.add_argument("--activations-dir", type=Path, default=Path("data/activations"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/ar"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--canvas-size", type=int, default=224)
    args = parser.parse_args()

    layer_dir = args.activations_dir / f"L{args.layer:02d}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"activations dir not found: {layer_dir}. Run activation_extractor.py first.")

    from safetensors.torch import load_file

    print(f"[ar-sup] loading activations from {layer_dir}", flush=True)
    h_all = load_file(layer_dir / "activations.safetensors")["h"].float()  # (N, hidden)
    texts: list[str] = []
    with open(layer_dir / "texts.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    print(f"[ar-sup] {h_all.shape[0]} activations, hidden={h_all.shape[1]}", flush=True)
    assert len(texts) == h_all.shape[0]

    # Load AV (frozen)
    print(f"[ar-sup] loading AV from {args.av_ckpt}", flush=True)
    av = StrokeDecoder.from_pretrained_and_extend(args.model_id, device="cuda", dtype=torch.bfloat16)
    # Restore vocab IDs from saved file (in case tokenizer order changed)
    saved_vocab = torch.load(args.av_ckpt / "stroke_vocab.pt", weights_only=False)
    from stroke_tokenizer import StrokeVocab
    av.vocab = StrokeVocab.from_name_to_id(saved_vocab["vocab_name_to_id"])
    # Apply LoRA from checkpoint
    from peft import PeftModel
    av.model = PeftModel.from_pretrained(av.model, args.av_ckpt)
    av.model.eval()
    for p in av.model.parameters():
        p.requires_grad = False

    # Build AR
    print(f"[ar-sup] building AR at layer {args.layer}", flush=True)
    ar = TruncatedGemmaAR.from_pretrained(args.model_id, layer_ell=args.layer, device="cuda")
    # AR's backbone is frozen; Linear head and (optional) LoRA trainable
    for p in ar.backbone.parameters():
        p.requires_grad = False
    for p in ar.linear.parameters():
        p.requires_grad = True
    # Add LoRA to AR's backbone
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    ar.backbone = get_peft_model(ar.backbone, lora_cfg)
    trainable = [p for p in ar.parameters() if p.requires_grad]
    print(f"[ar-sup] trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"L{args.layer:02d}_train.jsonl"
    t_start = time.time()

    # Cache: pre-generate one drawing per training example (frozen AV).
    # For Day-1 we sample drawings on the fly to save memory; in production
    # we'd cache them on disk.

    @torch.no_grad()
    def sample_drawing_for_text(text: str):
        # Use the text as a caption directly. If too long, take first 80 chars.
        caption = text[:80]
        # Use AV's batched SFT prompt with the caption, sample tokens
        # We don't have a "generate from caption" method, so build one inline.
        from av.stroke_decoder import SFT_PROMPT_TEMPLATE
        from av.activation_injection import stroke_token_ids
        from stroke_tokenizer import DRAW_CLOSE
        device = next(av.model.parameters()).device
        prompt = SFT_PROMPT_TEMPLATE.format(caption=caption)
        prompt_ids = av.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
        allowed = set(stroke_token_ids(av.vocab))
        gen_ids: list[int] = []
        past = None
        cur_ids = prompt_ids
        for _ in range(400):
            if past is None:
                out = av.model(input_ids=cur_ids, use_cache=True)
            else:
                out = av.model(input_ids=torch.tensor([[gen_ids[-1]]], device=device), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            mask = torch.full_like(logits, float("-inf"))
            allowed_ids = torch.tensor(list(allowed), device=device)
            mask[0, allowed_ids] = 0
            logits = logits + mask
            probs = torch.softmax(logits / 1.0, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            gen_ids.append(nxt)
            if nxt == av.vocab.name_to_id[DRAW_CLOSE]:
                break
        return gen_ids

    n = h_all.shape[0]
    perm = torch.randperm(n).tolist()
    cursor = 0

    with open(log_path, "a") as log_f:
        for step in range(args.steps):
            # Pull batch
            batch_idx = []
            while len(batch_idx) < args.batch_size:
                if cursor >= n:
                    perm = torch.randperm(n).tolist()
                    cursor = 0
                batch_idx.append(perm[cursor])
                cursor += 1

            images = []
            h_targets = []
            for idx in batch_idx:
                gen = sample_drawing_for_text(texts[idx])
                strokes = av.vocab.decode_tokens(gen)
                img = stroke_render(strokes, canvas_size=args.canvas_size)
                images.append(img)
                h_targets.append(h_all[idx])

            h_target = torch.stack(h_targets, dim=0).to("cuda")
            h_hat = ar.forward(images)  # (B, hidden)
            loss = F.mse_loss(h_hat.float(), h_target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                msg = {"step": step, "loss": float(loss.item()), "elapsed_sec": round(time.time() - t_start, 1)}
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()
                print(f"[ar-sup] step {step:5d} loss={msg['loss']:.4f}  ({msg['elapsed_sec']:.0f}s)", flush=True)

            if step % args.save_every == 0 and step > 0:
                save_dir = args.out_dir / f"L{args.layer:02d}" / f"step_{step:06d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                ar.backbone.save_pretrained(save_dir / "lora")
                torch.save(ar.linear.state_dict(), save_dir / "linear.pt")
                print(f"[ar-sup] saved → {save_dir}", flush=True)

    final = args.out_dir / f"L{args.layer:02d}" / "final"
    final.mkdir(parents=True, exist_ok=True)
    ar.backbone.save_pretrained(final / "lora")
    torch.save(ar.linear.state_dict(), final / "linear.pt")
    print(f"[ar-sup] DONE → {final}", flush=True)


if __name__ == "__main__":
    main()
