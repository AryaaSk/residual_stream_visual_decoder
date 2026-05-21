"""Build a single 12-concept × 7-layer giant grid video.
Plays all 84 hero animations simultaneously in a tiled layout."""
from __future__ import annotations

import subprocess
from pathlib import Path


LAYERS = ["L03", "L10", "L15", "L20", "L24", "L27", "L29"]
HERO = ["cat", "dog", "elephant", "horse", "fish", "bird",
         "sun", "flower", "tree", "mountain", "airplane", "car"]
IN = Path("artefacts/v3/viral/anim")
OUT = Path("artefacts/v3/viral/grid_anim.mp4")


def main():
    """Use a complex xstack filter to tile all 84 clips into a single grid."""
    inputs = []
    streams = []
    # Landscape orientation: layers as rows, concepts as columns (7 rows × 12 cols)
    n_cols = len(HERO)
    n_rows = len(LAYERS)
    n = 0
    tile_w = 180
    tile_h = 180
    for r, layer in enumerate(LAYERS):
        for c, concept in enumerate(HERO):
            p = IN / layer / f"{concept}.mp4"
            if not p.exists():
                # Use a blank
                continue
            inputs.extend(["-i", str(p)])
            streams.append((n, r, c))
            n += 1
    if n == 0:
        raise SystemExit("no inputs")

    # Build filter: scale each, place via xstack
    filter_lines = []
    for idx, (n_i, r, c) in enumerate(streams):
        filter_lines.append(
            f"[{n_i}:v]scale={tile_w}:{tile_h},setsar=1[v{n_i}]"
        )
    # xstack layout string: per-input "x_y"
    layout = []
    for n_i, r, c in streams:
        x = c * tile_w
        y = r * tile_h
        layout.append(f"{x}_{y}")
    layout_str = "|".join(layout)
    stream_in = "".join(f"[v{n_i}]" for n_i, _, _ in streams)
    filter_lines.append(
        f"{stream_in}xstack=inputs={len(streams)}:layout={layout_str}:"
        f"fill=white[out]"
    )
    flt = ";".join(filter_lines)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", flt,
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
        "-shortest",
        str(OUT),
    ]
    print(f"[grid-anim] {len(streams)} clips -> {OUT}")
    subprocess.run(cmd, check=True)
    print(f"[grid-anim] DONE")


if __name__ == "__main__":
    main()
