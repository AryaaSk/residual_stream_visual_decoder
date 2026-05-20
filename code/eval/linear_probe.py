"""Linear probe per layer — quantitative ceiling for v2.2.

For each of L3, L10, L20, L29 on Qwen 3.5-4B:
  1. Build a corpus of (concept, caption) examples. Concepts = the 44 categories
     the AV was SFT'd on (so we can compare to the visual decoder fairly).
     For each concept, use all caption templates from expanded_captions.jsonl.
  2. Extract h = Qwen(caption).hidden_states[layer][0, -1, :] for every (c, cap).
  3. Train sklearn LogisticRegression on a stratified 70/30 split by caption.
  4. Report top-1 and top-5 accuracy.

This is the BEST ANY DECODER (visual or otherwise) could extract from h at
each layer. It bounds the visual decoder's information-theoretic ceiling.

Output:
  findings/v2_2/probe_accuracy.json
  findings/v2_2/probe_accuracy.png    (bar chart layer vs accuracy)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# The 44 categories the AV was trained on (matches data/sft_quickdraw.jsonl).
SFT_CATEGORIES = [
    "airplane", "apple", "banana", "bed", "bicycle", "bird", "book", "bread",
    "bridge", "cactus", "car", "carrot", "cat", "chair", "clock", "cloud",
    "cookie", "dog", "donut", "door", "elephant", "fish", "flower", "horse",
    "house", "key", "leaf", "moon", "mountain", "mushroom", "pencil", "pizza",
    "rainbow", "scissors", "snake", "spider", "star", "sun", "table", "tent",
    "train", "tree", "truck", "umbrella",
]


def load_caption_templates(overlay_path: Path) -> dict[str, list[str]]:
    """Returns {concept: [caption_template_1, ...]} for concept_template rows."""
    out: dict[str, list[str]] = {}
    with open(overlay_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("source") != "concept_template":
                continue
            c = row.get("concept")
            cap = row.get("caption")
            if c and cap:
                out.setdefault(c, []).append(cap)
    return out


def extract_h_batch(model, tokenizer, captions: list[str], layers: list[int],
                    device: str = "cuda") -> dict[int, np.ndarray]:
    """For each layer, returns an (N, d) ndarray of last-token hidden states.

    Single forward pass per caption, picking the layers' hidden_states output.
    """
    layer_to_hs: dict[int, list[np.ndarray]] = {ell: [] for ell in layers}
    model.eval()
    for i, cap in enumerate(captions):
        enc = tokenizer(cap, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        for ell in layers:
            h = out.hidden_states[ell][0, -1, :].detach().to("cpu").to(torch.float32).numpy()
            layer_to_hs[ell].append(h)
        if (i + 1) % 50 == 0:
            print(f"[probe]   {i+1}/{len(captions)} captions extracted", flush=True)
    return {ell: np.stack(layer_to_hs[ell], axis=0) for ell in layers}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layers", type=int, nargs="+", default=[3, 10, 20, 29])
    p.add_argument("--captions-overlay", type=Path, default=Path("data/expanded_captions.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("findings/v2_2"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--C", type=float, default=1.0,
                   help="LogReg regularization (smaller = stronger)")
    p.add_argument("--max-iter", type=int, default=1000)
    args = p.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] loading Qwen base model {args.model_id} ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                  trust_remote_code=True,
                                                  torch_dtype=torch.bfloat16,
                                                  device_map="cuda").eval()

    print(f"[probe] loading caption templates from {args.captions_overlay} ...", flush=True)
    templates = load_caption_templates(args.captions_overlay)
    avail = [c for c in SFT_CATEGORIES if c in templates and len(templates[c]) >= 4]
    missing = [c for c in SFT_CATEGORIES if c not in templates]
    print(f"[probe] {len(avail)}/{len(SFT_CATEGORIES)} concepts have >=4 templates; missing={missing}", flush=True)

    # Build (concept, caption) pairs
    pairs: list[tuple[str, str]] = []
    for c in avail:
        for cap in templates[c]:
            pairs.append((c, cap))
    print(f"[probe] {len(pairs)} (concept, caption) pairs", flush=True)

    concept_to_idx = {c: i for i, c in enumerate(avail)}

    # Extract h at every requested layer (single forward pass per caption)
    t0 = time.time()
    captions = [cap for _, cap in pairs]
    layer_to_X = extract_h_batch(model, tok, captions, args.layers)
    y = np.array([concept_to_idx[c] for c, _ in pairs])
    print(f"[probe] extracted h in {time.time()-t0:.1f}s", flush=True)

    # Stratified split: for each concept, randomly choose 30% of its captions
    # for test. This guarantees every class appears in train and test.
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = [], []
    for c_idx in range(len(avail)):
        idxs = np.where(y == c_idx)[0]
        rng.shuffle(idxs)
        n_test = max(1, len(idxs) // 3)
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    print(f"[probe] split: {len(train_idx)} train / {len(test_idx)} test, n_classes={len(avail)}", flush=True)

    # Train + evaluate one probe per layer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    results = {
        "model_id": args.model_id,
        "n_concepts": len(avail),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "concepts": avail,
        "layers": args.layers,
        "per_layer": {},
    }
    for ell in args.layers:
        X = layer_to_X[ell]
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                                 n_jobs=-1, multi_class="multinomial",
                                 solver="lbfgs", random_state=args.seed)
        t1 = time.time()
        clf.fit(Xtr, y[train_idx])
        elapsed = time.time() - t1
        train_acc = float(clf.score(Xtr, y[train_idx]))
        test_acc = float(clf.score(Xte, y[test_idx]))
        # Top-5 accuracy on test
        probs = clf.predict_proba(Xte)
        top5 = float(np.mean([
            y[test_idx][i] in np.argsort(probs[i])[-5:]
            for i in range(len(test_idx))
        ]))
        # Per-concept accuracy
        per_concept = {}
        for c, ci in concept_to_idx.items():
            mask = y[test_idx] == ci
            if mask.sum() == 0:
                continue
            ypred = clf.predict(Xte[mask])
            per_concept[c] = {
                "n_test": int(mask.sum()),
                "accuracy": float(np.mean(ypred == ci)),
            }
        results["per_layer"][str(ell)] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "test_top5": top5,
            "fit_seconds": elapsed,
            "per_concept": per_concept,
            "h_norm_mean": float(np.linalg.norm(X, axis=1).mean()),
        }
        print(f"[probe] L{ell:02d}: train={train_acc*100:.1f}%  test={test_acc*100:.1f}%  top5={top5*100:.1f}%  ||h||~{results['per_layer'][str(ell)]['h_norm_mean']:.2f}  ({elapsed:.1f}s)", flush=True)

    out_json = args.out_dir / "probe_accuracy.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[probe] saved {out_json}", flush=True)

    # Bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        layers = args.layers
        accs = [results["per_layer"][str(ell)]["test_accuracy"] * 100 for ell in layers]
        top5s = [results["per_layer"][str(ell)]["test_top5"] * 100 for ell in layers]
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(layers))
        w = 0.35
        ax.bar(x - w/2, accs, w, label="top-1 (test)", color="#3066be")
        ax.bar(x + w/2, top5s, w, label="top-5 (test)", color="#9bc1ff")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{ell}" for ell in layers])
        ax.set_ylabel("Probe accuracy (%)")
        ax.set_ylim(0, 100)
        ax.axhline(100 / len(avail), color="grey", linestyle="--",
                   label=f"chance ({100/len(avail):.1f}%)")
        ax.set_title(f"Linear probe on Qwen 3.5-4B last-token h\n({len(avail)} concepts, {len(pairs)} (concept,caption) pairs)")
        ax.legend()
        for i, (a, b) in enumerate(zip(accs, top5s)):
            ax.text(x[i] - w/2, a + 1, f"{a:.0f}", ha="center", fontsize=9)
            ax.text(x[i] + w/2, b + 1, f"{b:.0f}", ha="center", fontsize=9)
        plt.tight_layout()
        out_png = args.out_dir / "probe_accuracy.png"
        plt.savefig(out_png, dpi=120)
        print(f"[probe] saved {out_png}", flush=True)
    except Exception as e:
        print(f"[probe] WARN: plot failed: {e}", flush=True)


if __name__ == "__main__":
    main()
