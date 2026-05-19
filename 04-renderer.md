# 04 — Renderer + animation

The renderer is a **deterministic, no-learned-params** Python module that turns AV's stroke token sequence into a 224×224 PNG, while simultaneously accumulating intermediate frames for animation.

## Two outputs from one pass

```
   AV stroke tokens
        │
        ▼
  ┌────────────┐
  │ RENDERER   │── (a) final PNG (224×224)    ──→ fed to AR (used for training/inference)
  │            │── (b) MP4 / GIF              ──→ for human inspection only
  └────────────┘
```

The final PNG is the only artefact the AR sees. The animation is for us. Both come for free from a single rendering pass because the renderer is sequential.

## Why animation matters

Three reasons:

1. **Order is signal.** A drawing's stroke order reveals what the model "decides first". Does it lay down the rough outline before details? Does it draw eyes before the face? Does it jump around (suggesting parallel features) or proceed left-to-right (suggesting sequential reasoning)? This is interpretability content that a static raster cannot convey.

2. **Stroke order is what raster generators cannot do.** Chameleon, Anole, Transfusion, GPT-4o native image-gen all produce rasters in one shot. The temporal structure is missing. Our stroke representation gets it for free.

3. **Demos.** A 5-second MP4 showing Gemma 4 "drawing its thoughts" frame by frame is a thousand times more visceral than a static image grid. For papers, talks, and intuition-building, this is the artefact.

## Pseudocode

```python
def render(stroke_tokens, canvas_size=224, save_animation_path=None):
    canvas = blank_grayscale(canvas_size, fill=255)   # white background
    pen_pos = (canvas_size // 2, canvas_size // 2)     # start at centre
    pen_down = False
    frames = []                                        # for animation

    for triplet in chunk_into_triples(stroke_tokens):
        dx, dy, pen_state = decode(triplet)
        new_pos = (pen_pos[0] + dx, pen_pos[1] + dy)

        if pen_down:
            draw_antialiased_line(canvas, pen_pos, new_pos, width=2)

        pen_pos = new_pos

        if pen_state == DOWN:
            pen_down = True
        elif pen_state == UP:
            pen_down = False
        elif pen_state == END:
            break

        if save_animation_path is not None:
            frames.append(canvas.copy())

    if save_animation_path is not None:
        write_video(frames, save_animation_path, fps=24)

    return canvas
```

That is the whole renderer. PIL + numpy is enough; Cairo is nicer for anti-aliasing.

## Animation format choices

- **MP4 (h.264)**: best for embedding in papers, small file size, smooth playback. Default.
- **GIF**: best for sharing in Slack/Twitter/blog posts. Larger file, lossier colour.
- **APNG**: best fidelity, niche viewer support.

Write all three on the eval pipeline. Use MP4 for batch generation during training (cheaper to store).

## Implementation notes

- **Pen starts at canvas centre** (112, 112) so the model can draw in any direction from the start.
- **Coordinate range:** `(Δx, Δy) ∈ [-64, 63]` per stroke, quantised to 128 bins. Cumulative pen position clamped to `[0, 223]` so we don't draw off-canvas.
- **Line width: 2 px** with anti-aliasing for visual smoothness. Bigger lines hide stroke ordering; thinner lines look harsh.
- **Background: white, ink: black.** Matches QuickDraw and IAM On-Line conventions; ensures the rendered image looks like a sketch the vision encoder might have seen something like.
- **No colour for v0.** Colour-aware extension is a clean v2: add `color_R_bin`, `color_G_bin`, `color_B_bin` tokens (3 extra tokens per stroke). Quadruples vocab but each token still 1-d.
- **Optional: variable line width.** Add `width_bin` token. Lets the model express emphasis. Defer to v2.

## Renderer is also the eval surface

Every generated drawing during training and eval goes through this renderer. So:
- It must be **fast**: <10 ms per drawing on CPU.
- It must be **deterministic**: same tokens → same PNG, bit-exact. Required for reproducible eval.
- It must **handle malformed sequences gracefully**: AV may emit `Δx_bin` twice in a row before `Δy_bin`. Skip or pad with a neutral value rather than crash. Log malformations as a training metric.

## Frame-rate decision

Animation FPS is purely a viewing choice. Sensible options:
- **24 fps**: filmic, smooth, every stroke gets ~40 ms.
- **60 fps**: hi-fi, every stroke gets ~17 ms.
- **N strokes per second** where N is chosen to make a 200-stroke drawing land at ~5 sec runtime. `N = 40` → 5 sec.

Default: **24 fps with one frame per stroke**, so drawing length scales naturally with content complexity.
