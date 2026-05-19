# Morning checklist — pulling v1.1 results

Written 2026-05-19 23:05 UTC, before going to sleep. Everything below assumes
the autonomous pipeline ran overnight without you needing to babysit it.

## Quick status check (run first)

```bash
cd /Users/aryaask/Desktop/residual_stream_visual_decoder
ssh -i ~/.ssh/gcp_zoral_h200_ed25519 theod@35.230.182.229 'tmux ls; echo ---; cd /home/theod/Aryaa/rsvd && ls findings/v1_1/ && ls artefacts/v1_1/ 2>/dev/null && echo --- && tail -30 runs/autofinish.log'
```

You should see:
- `autofinish:` tmux session either still running (rare — training was slow) or gone (good — completed)
- `findings/v1_1/SUMMARY.json` present → entire pipeline finished
- `artefacts/v1_1/demo.mp4` present → hype reel made

## Pull everything

```bash
cd /Users/aryaask/Desktop/residual_stream_visual_decoder
bin/h200 pull checkpoints/v1_1/
bin/h200 pull findings/v1_1/
bin/h200 pull artefacts/v1_1/
bin/h200 pull runs/autofinish.log
bin/h200 pull runs/iter11L12.log
bin/h200 pull runs/iter11L24.log
```

## Read the headline number

```bash
cat findings/v1_1/SUMMARY.json | python3 -m json.tool | head -80
```

Key fields:
- `layers.L12.heldout.fve` and `layers.L24.heldout.fve` — the headline FVE
  - **Target**: > 0.10 (v0.1 + v1.0 were ~0)
  - **OK**: 0.03 – 0.10 (real signal, modest)
  - **Bad**: < 0.03 (still broken, ship as honest negative result)
- `layers.L*.train_dist.fve` — diagnostic; should be higher than heldout
- `layers.L*.per_iteration` — should show FVE climbing across iterations

## Look at the visuals (in priority order)

1. `artefacts/v1_1/demo.mp4` — 60-sec hype reel. Open in QuickTime.
2. `findings/v1_1/inject_demo_L12/*.mp4` — drawings per hero prompt at L12
3. `findings/v1_1/inject_demo_L24/*.mp4` — same at L24
4. `artefacts/v1_1/cross_layer/*.mp4` — same prompt across L12 and L24
5. `artefacts/v1_1/trajectory_L12/*.mp4` — drawing morph per generated token

## Pareto judgement call

After looking at numbers + visuals, decide:

- **FVE > 0.10 + visibly correlated drawings** → ship as v1.0 / v1.1 release.
  Fill in WRITEUP.md placeholders with actual numbers, commit, tag `v1.1`,
  `gh release create v1.1` with `demo.mp4` attached. **Then tweet.**

- **FVE > 0.03 but visuals don't clearly correspond** → ship as honest
  negative-result writeup. Title: "Trying to draw what Gemma 4 thinks: 6
  months of failure modes." This is still viral — interpretability twitter
  loves honest postmortems.

- **FVE still ~0** → consider trying:
  - even later layer (L32?) — L24 is mid-late, L32 is the last interesting one
  - alpha sweep (we locked 0.5; maybe 0.7 or 1.0 helps at later layers)
  - bigger AR LoRA (rank 32 instead of 16)
  - more iterations (we ran 4; NLA paper ran ~10)

## Update the writeup

The WRITEUP.md has placeholders like `{L12_FVE}` and `{L24_FVE}`. Find them
and substitute the real numbers.

```bash
grep -n "{L" WRITEUP.md
```

## Commit + push + tag

```bash
git add findings/v1_1/ artefacts/v1_1/ WRITEUP.md README.md
git commit -m "v1.1 final results: L12 FVE=X, L24 FVE=Y, full eval artefacts"
git push origin master
git tag v1.1 -a -m "v1.1: faithful visual decoder, FVE=X (held-out)"
git push origin v1.1
gh release create v1.1 artefacts/v1_1/demo.mp4 --notes-file findings/v1_1/RELEASE_NOTES.md
```

## If autofinish hung

If the autofinish tmux session is still running > 12 hr after launch, attach
and see what step it's on:

```bash
ssh -i ~/.ssh/gcp_zoral_h200_ed25519 theod@35.230.182.229 -t 'tmux attach -t autofinish'
# (Ctrl-B then D to detach)
```

The script is idempotent — kill and re-run if needed:

```bash
ssh ... 'tmux kill-session -t autofinish; tmux new-session -d -s autofinish "bash /home/theod/Aryaa/rsvd/bin/autofinish.sh"'
```

## What the autofinish does (for context)

In sequence after both training jobs hit DONE:
1. `inject_demo` for L12 and L24 (PNG + MP4 per hero prompt)
2. `measure_fve` on held-out probes for each layer
3. `measure_fve_train_dist` on training-distribution probes (sanity diagnostic)
4. `plot_iter_log` (FVE-per-iteration plot per layer)
5. `cross_layer_trajectory` (L12 + L24 across-depth MP4)
6. `token_trajectory` (per-token morph MP4 for L12 only)
7. `make_hype_reel` (60-sec demo.mp4)
8. Writes `findings/v1_1/SUMMARY.json` consolidating everything

All under `set +e` so individual failures don't block the whole pipeline.
