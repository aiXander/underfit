# Plan: first SA3 LoRA finetune on our own audio

**Status:** researched, nothing run yet. Goal: a DoRA adapter on SA3-medium that carries one style from our own audio, usable at inference on the Mac. The upstream [README](../../README.md) is the full manual for the dashboard. This file keeps only our decisions, findings and open steps.

**Where things run:** prep and inference on the M2 Mac. **Training always runs on the NVIDIA GPU box** (the Mac is too slow; Xander's call). Modal only if the box can't do it, and that needs a go-ahead.

## Open: next steps

1. **Pick the dataset (Xander).** 30–90 min of *one* coherent style. Source audio lives in `AI_Audio/audio-data/<artist>/<release>/` (gitignored, never commit).
2. **Train on the GPU box** via the dashboard: `./install.sh && ./run.sh` → http://localhost:8787 (ssh-tunnel it). First accept the SA3 license once on HF (https://huggingface.co/stabilityai/stable-audio-3-medium), otherwise the ARC downloads fail with 401. Point `HF_HUB_CACHE` at `AI_Audio/models/hf-cache` (repo rule: all downloads go into `models/`). The medium pack is ~17 GB. Encode the dataset in the dashboard (**+ Dataset**), then **+ Finetune** with the config below.
3. **Pick checkpoints by ear** (see below), copy the chosen `.safetensors` to the Mac (`AI_Audio/models/derived/`), and render with `sa3lab/sa3 --dit medium --decoder same-l --lora <ckpt>`.
4. **Log** the dataset, config, chosen step and how it sounds in the SA3 research Claude Doc (URL in `AI_Audio/CLAUDE.md`).

## Findings

### Data
- **10 min is the floor; 30+ min is better.** Past that, quality beats quantity.
- **One style per dataset.** Mixed material trains into mush. For several styles, train one LoRA each and stack them at inference (`--lora a --lora b`).
- Any format/sample rate/channel count works: it gets resampled to 44.1 kHz stereo. Full songs are fine, because training takes random crops.
- Latents are codec-locked: **medium ↔ SAME-L, small models ↔ SAME-S.** Re-encode when switching model families.

### Captions are not needed
- dadabots are right. The README explicitly supports "skip metadata entirely". Why it works:
  - 10% of steps drop the prompt anyway (CFG dropout), so the style is always partly learned unconditionally.
  - Late in a run even empty-prompt demos take on the style. The README calls this the "unconditional crossover".
- **Our default:** a fixed made-up trigger string (e.g. `xndrsound`) on every clip. It gives an inference handle that combines with SA3's own `Genre: …, BPM: …` vocabulary.
- Free extra prompt sources, no hand captioning:
  - embedded ID3/FLAC tags
  - folder/file paths (meaningful subfolders become prompts)
  - `.txt`/`.json` sidecars
- The README suggests **~50% fixed trigger + 50% tags/paths**. `dataset_processing/autotagger.py` writes JSON sidecars from `Artist - Album (Year)/NN Title` folder layouts.

### Run config (first run)

| Knob | Value | Why |
|---|---|---|
| Model | `sa3-medium` | best quality; train on Base, LoRA also works on ARC |
| Adapter | DoRA-rows, rank 16 | dadabots' default; column variants are deprecated |
| Latent crop | **~512 (≈47 s), random crop on** | learns style without memorising song structure; ~128 (≈12 s) is the even-shorter option |
| Steps | 10k–20k; checkpoint every 500, demos every 500–1000 | ~10k = "creatively underfit"; past 20k it starts memorising |
| Batch | 1–4 | creative knob: 1 = sharper per-song imprint, 4 = smoother average |
| LR | default (1e-4) | |

Latent rate is 44100/4096 ≈ 10.77 latents/s, so the medium default crop of 4096 ≈ 380 s, i.e. whole songs.

### Choosing a checkpoint
- The **loss-curve elbow** (where loss first starts dropping) is usually the most interesting checkpoint.
- Signs a run has got it:
  1. CFG 7 demos sound right first.
  2. Later, CFG 1 sounds cleaner than CFG 7.
  3. Empty-prompt demos take on the style.
- ARC demos lag but end up cleaner.
- Keep several checkpoints: memorised ones are still useful for audio-to-audio style transfer.

### Inference
- **Strength 0.6–0.8** gives "in the style of" without regurgitation.
- Skipping the LoRA on the first denoising step lets the base model set the song structure.
- Resuming from a LoRA on a second dataset (e.g. 10k steps at 47 s, then 300 steps at 12 s on another style) fuses the two styles.

## Known issues / unverified

- **Mac-only trap (irrelevant on the GPU box):** with `UNDERFIT_ENGINE=mlx`, the dashboard's prompt config is silently dropped. That covers the trigger word, fixed text and path prompts.
  - Cause: `underfit/backends/mlx_engine.py` `build_trainer_cmd` never passes `--prompt-config` to the MLX trainer.
  - Effect: the MLX trainer falls back to plain tag prompts, or to an empty prompt when there are no tags, i.e. training is unconditional.
  - Fix if we ever train via MLX: write `dataset_config["prompt_config"]` to a temp JSON and pass it as `--prompt-config`.
- **Unverified:** the MLX pre-encoder (`stable-audio-3/optimized/mlx/scripts/pre_encode_mlx.py`) claims to write underfit's exact latent format, so encoding on the Mac and uploading only latents (~40 MB per hour of audio) should work. It isn't tested against the torch trainer. Encoding on the GPU box is the safe default.
