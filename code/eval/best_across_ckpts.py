"""Pick the single best CLIP-ranked drawing per concept across all checkpoints.

For each concept, scan every findings/v*/clip_L*_*/summary.json (which contains
the CLIP scores per candidate per concept), find the checkpoint with the
highest top-1 score, and copy the top-K drawings out into a "best of best"
directory.

This is the curation step before building the final hype reel.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--findings-roots", type=Path, nargs="+",
                   default=[Path("findings/v1_2"), Path("findings/v1_3")])
    p.add_argument("--out", type=Path, default=Path("artefacts/v1_3/best_of_best"))
    p.add_argument("--copy-mp4", action="store_true", default=True)
    args = p.parse_args()

    # Gather all clip_* subdirs that have a summary.json
    candidate_dirs = []
    for root in args.findings_roots:
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir() or not d.name.startswith("clip_"):
                continue
            sj = d / "summary.json"
            if sj.exists():
                candidate_dirs.append(d)
    print(f"[best] {len(candidate_dirs)} CLIP-ranker output dirs found")

    # Per concept, find best (dir, score, slug-suffix)
    best_per_slug: dict[str, dict] = {}
    for d in candidate_dirs:
        try:
            summary = json.loads((d / "summary.json").read_text())
        except Exception:
            continue
        for row in summary:
            slug = row.get("slug")
            score = row.get("best_clip", -1e9)
            if slug is None:
                continue
            prev = best_per_slug.get(slug)
            if prev is None or score > prev["score"]:
                best_per_slug[slug] = {"score": score, "dir": d}

    # Copy best out
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for slug, info in sorted(best_per_slug.items()):
        src_dir = info["dir"]
        src_png = src_dir / f"{slug}_top0.png"
        if not src_png.exists():
            # fallback for older single-sample format
            src_png = src_dir / f"{slug}.png"
        if not src_png.exists():
            print(f"[best] skip {slug}: no png in {src_dir}")
            continue
        dst_png = args.out / f"{slug}.png"
        shutil.copy(src_png, dst_png)
        if args.copy_mp4:
            src_mp4 = src_png.with_suffix(".mp4")
            if src_mp4.exists():
                shutil.copy(src_mp4, args.out / f"{slug}.mp4")
        print(f"[best] {slug:10s} CLIP={info['score']:.2f}  from {src_dir.name}")
        manifest.append({"slug": slug, "score": info["score"], "from": src_dir.name})

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[best] wrote {len(manifest)} drawings to {args.out}")


if __name__ == "__main__":
    main()
