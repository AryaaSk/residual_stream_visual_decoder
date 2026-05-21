"""Locally compose a per-concept cross-layer side-by-side animation.

For each hero concept, take the per-layer MP4s already on disk:
    artefacts/v3/viral/anim/L{NN}/{concept}.mp4
and hstack them into one frame-synchronised 7-panel video.

Output: artefacts/v3/viral/cross_layer_anim/{concept}.mp4 (7 panels)
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

LAYERS = ["L03", "L10", "L15", "L20", "L24", "L27", "L29"]

HERO = ["cat", "elephant", "sun", "dog", "horse", "flower", "car", "fish",
        "tree", "airplane", "bird", "mountain"]


def make_cross(concept: str, in_dir: Path, out_path: Path, height: int = 320) -> bool:
    inputs = []
    filter_parts = []
    n = 0
    for L in LAYERS:
        p = in_dir / L / f"{concept}.mp4"
        if not p.exists():
            continue
        inputs.extend(["-i", str(p)])
        filter_parts.append(
            f"[{n}:v]scale=-2:{height},pad=iw+2:ih+60:0:60:white,"
            f"drawtext=text='{L}':fontcolor=black:fontsize=28:x=12:y=12[v{n}]"
        )
        n += 1
    if n == 0:
        return False
    hstack_in = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{hstack_in}hstack=inputs={n}[out]")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", "24", "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=Path("artefacts/v3/viral/anim"))
    p.add_argument("--out-dir", type=Path, default=Path("artefacts/v3/viral/cross_layer_anim"))
    p.add_argument("--height", type=int, default=320)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for c in HERO:
        out = args.out_dir / f"{c}.mp4"
        ok = make_cross(c, args.in_dir, out, height=args.height)
        print(f"[xlayer-anim] {c:12s}  {'ok' if ok else 'missing'}  → {out}")


if __name__ == "__main__":
    main()
