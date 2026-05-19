"""Assemble the 60-second hype reel for v1.0 release.

Takes a list of MP4 paths + titles, stitches them together with fade-ins
and title cards, normalises resolution and frame rate. Output is a single
MP4 suitable for embedding in tweets / blog posts / GitHub README.

Layout per clip:
  - 1-second title card (centred caption on white background)
  - 4-second clip (with crossfade in)
  - tiny gap

If a clip is shorter than 4 seconds, it's looped to fill the duration.

Usage:
    python code/eval/make_hype_reel.py --in-dir artefacts/trajectory --out demo.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ffmpeg_concat(clip_paths: list[Path], out_path: Path) -> None:
    """Use ffmpeg's concat demuxer to join clips. Assumes uniform codec/resolution."""
    list_file = out_path.parent / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in clip_paths))
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(out_path),
    ], check=True)
    list_file.unlink(missing_ok=True)


def make_title_card(text: str, duration: float, size: tuple[int, int], out_path: Path,
                    fps: int = 30) -> None:
    """Render a 1-second title card via ffmpeg's drawtext + color."""
    W, H = size
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=white:s={W}x{H}:r={fps}:d={duration}",
        "-vf",
        f"drawtext=text='{text.replace(chr(39), chr(96))}':"
        f"fontfile=/System/Library/Fonts/Helvetica.ttc:"
        f"fontsize=36:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path),
    ]
    subprocess.run(cmd, check=True)


def normalise_clip(in_path: Path, out_path: Path, *, size: tuple[int, int], fps: int = 30,
                   target_duration: float = 4.0) -> None:
    """Scale/pad clip to standard size, set framerate, loop or trim to target_duration."""
    W, H = size
    subprocess.run([
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(in_path),
        "-t", str(target_duration),
        "-vf",
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,"
        f"fps={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        str(out_path),
    ], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=Path("artefacts/trajectory"),
                   help="Directory with input MP4s (one per clip)")
    p.add_argument("--out", type=Path, default=Path("demo.mp4"))
    p.add_argument("--width", type=int, default=896)
    p.add_argument("--height", type=int, default=896)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--title-duration", type=float, default=1.0)
    p.add_argument("--clip-duration", type=float, default=4.0)
    p.add_argument("--max-clips", type=int, default=12)
    p.add_argument("--captions", type=Path, default=None,
                   help="Optional JSON file mapping slug → caption")
    args = p.parse_args()

    clips = sorted(args.in_dir.glob("*.mp4"))
    if not clips:
        print(f"[hype] no clips in {args.in_dir}", file=sys.stderr)
        sys.exit(1)
    clips = clips[: args.max_clips]

    import json
    captions: dict[str, str] = {}
    if args.captions and args.captions.exists():
        captions = json.loads(args.captions.read_text())

    work = args.out.parent / ".hype_workdir"
    work.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    print(f"[hype] preparing {len(clips)} clips")
    for i, clip in enumerate(clips):
        slug = clip.stem
        caption = captions.get(slug, slug.replace("_", " "))
        title_path = work / f"{i:02d}_title.mp4"
        norm_path = work / f"{i:02d}_clip.mp4"
        make_title_card(caption, args.title_duration, (args.width, args.height),
                        title_path, fps=args.fps)
        normalise_clip(clip, norm_path, size=(args.width, args.height), fps=args.fps,
                       target_duration=args.clip_duration)
        segments.append(title_path)
        segments.append(norm_path)
        print(f"  [hype] segment {i}: {slug}")

    # Concat all segments
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_concat(segments, args.out)
    # Clean up workdir
    for f in work.iterdir():
        try:
            f.unlink()
        except Exception:
            pass
    try:
        work.rmdir()
    except Exception:
        pass

    print(f"[hype] wrote {args.out}")


if __name__ == "__main__":
    main()
