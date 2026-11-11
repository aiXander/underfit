# Djrum v1 LoRA (`djrum-uts-v1`): what's left after training

**For the agent picking this up:** training is finished; what remains is Xander's listening and the Mac-side follow-through below. How to launch and babysit a run, the standard demo set and how to read the dashboard charts are in [../reference/sa3-lora-training.md](../reference/sa3-lora-training.md) (**Running a job**, **Choosing a checkpoint**). Paths are relative to `AI_Audio/`.

## Open

1. **Xander listens** to the step × demo grid in `outputs/underfit-djrum-uts-v1/` (8 steps: 0 = no-LoRA baseline, then 2k … 14k; 12 demos each, demo rows explained in the reference doc's standard demo set). Most useful notes: the step where d0 (training prompt) first sounds like the album, and the first step where each genre row d4–d9 flips toward Djrum (collapse point = last checkpoint that still generalises).
2. **Shortlist 2–3 checkpoints** from that. Copy them from the GPU box's `underfit/state/runs/djrum-uts-v1-20261105174615/403ebe6b/checkpoints/djrum-uts-v1-step=<N>-epoch=<E>.safetensors` to the Mac's `AI_Audio/models/derived/djrum-uts-v1/` (Xander moves files between machines).
3. **Render on the Mac** with ARC at 120 s, tag-style prompts, two seeds per prompt, strengths 0.6 / 0.8 / 1.0. **No trigger word** (this run trained without one), so prompts are just `TrackType: Music, VocalType: Instrumental, …`:
   `sa3lab/sa3 --dit medium --decoder same-l --seconds 120 --prompt "TrackType: Music, VocalType: Instrumental, ..." --lora <ckpt> strength=0.8 steps=2- --out <abs path in outputs/>` (`steps=2-` skips the LoRA on the first denoising step so the base model sets the structure; `--seconds` defaults to 30, always pass 120).
4. **Log** dataset, config, chosen step(s) and how they sound in the SA3 research Claude Doc's Log table (URL in `AI_Audio/CLAUDE.md`).

## What ran (2026-11-05)

DoRA-rows r16 on sa3-medium-base (fp32), LR 1e-4, batch 1, 512-latent (~47 s) random crops, on Djrum's *Under Tangled Silence* (11 tracks, 62 min, dataset `djrum-uts-sa3-medium-20261105`). Config: `sa3lab/train/djrum_uts_v1.json`; the only change from the verified 20-step check was **removing the trigger word** (Xander), so every clip trained on `TrackType: Music, VocalType: Instrumental` (10% empty). 14k steps, checkpoint + 12 demos every 2k, 17:46–21:05, no incidents. Mean loss per 2k window fell the whole way (0.82 → 0.72), most of it at high noise (σ 0.8–1.0: 0.92 → 0.80), the signature of learning the album's specific passages; each track was seen ~1,270 times, so expect the late checkpoints to memorise. Low-noise loss stayed flat.

## Open questions / v2 candidates

- **No validation loss**: hold out one track in v2 so a train-vs-val gap can actually show overfitting.
- **Trigger or not**: v1 has none; if the style leaks into every prefixed prompt, a triggered run is the comparison.
- Tracks over 600 s are truncated at encode (*Sycamore* loses its last 68 s).
- Short tracks get the same sampling weight as long ones (one item per file). If *Hold* (111 s) or *Reprise* (156 s) dominate the style, chop the album into equal segments before encoding.
- Only if v1 is underwhelming: 128-latent (~12 s) crop; batch 2; 50/50 fixed + hand-written per-track `Genre/Mood/BPM` sidecars (next to the audio, not in a `json/` subfolder).
