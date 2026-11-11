# SA3 LoRA training with underfit — how it works here

How our fork of underfit trains SA3 LoRAs on the GPU box (rednax-beast, RTX 3090 24 GB, Ubuntu): setup, what the pipeline actually does with our data, the evidence behind the knobs, and the traps. The upstream [README](../../README.md) is the dashboard manual; this file only holds what the README doesn't say or gets wrong for us. How an agent launches and babysits a run is in **Running a job** below. Open work on the Djrum v1 run (listening, checkpoint shortlist, Mac renders) lives in [../TODO/sa3-lora-finetune.md](../TODO/sa3-lora-finetune.md).

## Setup on the GPU box

- **Env:** `uv sync --inexact` in `underfit/` (torch 2.7.1+cu128, py3.10), then the setup wizard `uv`-installs the sibling `../stable-audio-3` editable (`[lora,ui]` extras) into the same `.venv`. Non-interactive form:
  `HF_HUB_CACHE=$AI_Audio/models/hf-cache .venv/bin/python -m underfit.cli.setup --backend sa3 --backend-path ../stable-audio-3 --models sa3-medium`
  (it also needs `HF_TOKEN` exported — Xander's is in `~/12TBHDD/Github/ai-toolkit/.env`; the account has accepted the SA3 licence.)
- **Weights:** `sa3-medium` = `stable-audio-3-medium-base` (14.3 GB: DiT 9.2 GB + `svd_bases.pt` 3.8 GB for the -XS adapters + T5Gemma 1.2 GB) + `stable-audio-3-medium` ARC (10.5 GB). They live in `AI_Audio/models/hf-cache/` (repo rule); `state/models/sa3-medium/{base,arc}` are symlinks into it. The SAME-L codec is inside the DiT checkpoint — nothing else to download. The small packs are not downloaded; the dashboard's "HF cache empty for …small…" startup lines are harmless.
- **Dashboard:** `setsid nohup env UNDERFIT_DASHBOARD_HOST=127.0.0.1 HF_HUB_CACHE=… ./run.sh > state/dashboard.log 2>&1 < /dev/null & disown` → http://127.0.0.1:8787. Trainers are spawned with `os.setsid` (`_popen_managed` in `server.py`) and survive a dashboard restart, which re-attaches them from `state/runs.json`. GPU history in the UI is VRAM only, last hour, in memory. **Default host is `0.0.0.0`** (open to the LAN) — always pin it to localhost. From the Mac: `ssh -L 8787:127.0.0.1:8787 <gpu-box>`.
- **xet downloads buffer in RAM** before hitting disk (the cache dir looks stuck at a few MB while the NIC does ~60 MiB/s and RSS grows to ~6 GB). Not a hang.
- `flash-attn` is not installed; training uses PyTorch FlexAttention (first steps pause for triton autotune). Fine on Ampere at batch 1; only worth installing for batch > 1.

## What the pipeline does with our data

- **Encode** (`dataset_processing/pre_encode.py`, launched by **+ Dataset**): each file → one `.npy` latent `[256 × 6459]` + `.json` sidecar. Every file is **padded or truncated to 600 s** (`max_samples = 600*sr`); the sidecar's `padding_mask` marks real frames. Tracks over 10 min lose their tail — split them first if the tail matters. 62 min of audio encodes in ~1 min on the 3090 → 70 MB.
- **Crop** (`stable_audio_3/data/dataset.py` `PreEncodedDataset`): with `random_crop` the start is drawn from `[0, last_real_frame − crop]`, so crops never land in padding. One dataset item per *file*, so short tracks are sampled as often as long ones (a 111 s track gets the same weight as a 600 s one).
- **`seconds_total`** conditioning gets the *whole track's* length, not the crop's. A 47 s crop is labelled e.g. "353 s". Another reason to keep the `seconds_total` conditioner out of the LoRA (below).
- **Prompts** (`dataset_processing/prompt_templates.py`): per step, one source is picked by the balance weights (tags / paths / fixed text), then the trigger word is prepended with `trigger_pct` probability. CFG dropout (empty prompt) is 10%, same as SA3 pretraining.
- **Layers:** the **+ Finetune** form's default layer filter trains only the 24 transformer blocks (self-attn, cross-attn, ff → 168 LoRA layers, 19.1 M params at r16) and excludes every conditioning embedder (incl. `seconds_total`) plus `project_in/out` and `pre/postprocess_conv` (`_applyDefaultExclusions` in `dashboard/index.html`). **The raw `/api/runs/new` API has no such default** — a scripted launch must pass `lora_exclude` itself, or it trains all 229 layers. `AI_Audio/sa3lab/train/djrum_uts_v1.json` carries the exact UI list.
- **LR schedule:** AdamW (β 0.9/0.95, wd 0.01) + `InverseLR` with exponential warmup 0.995 → LR reaches ~40% of target by step 100, ~92% by step 500, full by ~1k. Decay is negligible (`inv_gamma` 1e6). So nothing much happens in the first few hundred steps.
- **Precision:** autocast fp16 with grad scaler (`--precision 16-mixed`, hardcoded in the launch command), LoRA weights fp32. The frozen base is cast only if `base_precision` is set: the **form defaults it to fp16**, the API (no key) keeps the shipped **fp32** weights. fp32 base measured no slower and still ~7 GB VRAM, so we use it for quality runs.
- **Demos** pause training every `demo_every` steps and render each `demo_cond` entry (RF = base model, 50 steps; ARC = the 8-step model swapped in, then base weights restored). `duration` per entry is in seconds and may exceed the training crop.

## Measured on the 3090 (sa3-medium, DoRA-rows r16, batch 1, crop 512)

| | |
|---|---|
| Model load + compile | ~1 min |
| Training | ~1.25 steps/s (fp16 or fp32 base alike). `djrum-uts-v1`: 14k steps + 8 demo rounds = **3 h 18 min** wall clock |
| VRAM / GPU | ~7.0 GB (lots of headroom); ~80 % average util, ~320 W, max 84 °C (no throttling) |
| Demo, RF 50 steps | 20 s ≈ 10 s; 120 s ≈ 15 s. ARC 8 steps at 120 s ≈ 5 s. Our 12-demo set at 120 s ≈ 80 s per round |
| Checkpoint | ~45 MB `.safetensors` at `state/runs/<run>/<hash>/checkpoints/` |

Shell control of the dashboard (launch from a payload JSON, status, copy demos to `outputs/` with descriptive names): `AI_Audio/sa3lab/train/uf.py` (docstring).

## Why the knobs are set the way they are

Community material on SA3 LoRAs is thin (Nov 2026: one public adapter on HF, empty discussion tabs, no Reddit threads with numbers). Evidence by source:

- **Steps.** Stability's own `docs/workflows/lora.md` (in `../stable-audio-3`) starts at **1k steps**, rank 16, DoRA-rows, and calls it "a starting point". fal's hosted SA3 trainer also defaults to 1k (max 20k). dadabots say ~10k is where it "creatively underfits", >20k memorises. Our default: run 10k but **start listening at 500–1k** — the official default implies something usable may appear early. 62 min in 47 s crops ≈ 80 windows → 10k steps ≈ 125 passes.
- **Exclude `seconds_total`.** Stability's lora.md: "prevents conditioner hijacking on small datasets". The dashboard form already excludes it (with all conditioning, see above).
- **LR 1e-4, rank 16, alpha = rank** is the default everywhere (Stability, fal, underfit). An ACE-Step rank-64 run at 1e-4 diverged — another model, but a reason not to raise rank and LR together.
- **Crop 512 latents (~47 s), random.** Long enough to carry the style, short enough not to memorise whole song structures; crops never land in padding (see Crop above).
- **Batch 1–2.** VRAM allows much more; nobody reports it helping, and larger batch/rank pushes toward memorisation (Pinokio guide, single source).
- **Prompt format.** SA3 pretraining prompts were random subsets of metadata fields, field names included half the time (`Genres: …, Moods: …`), shuffled half the time, lowercased half the time; the paper says `TrackType: Music, VocalType: Instrumental,` prefixes "significantly improve generation quality" (see `../stable-audio-3/docs/reference/sa3-model-map.md`). So a fixed `TrackType: Music, VocalType: Instrumental` is in-distribution. **`djrum-uts-v1` trained with no trigger word** (Xander's call, `trigger: ""`, `trigger_pct: 0`): the style can only bind to that prefix and to the empty prompt, so any tag prompt starting with the prefix inherits it, and genre-diversity demos may drift toward the dataset sooner than with a trigger. Whether a trigger is worth having is still open; compare against a triggered run before settling it.
- **Our own blind bench (2026-11-05, SA3-medium on MLX, no LoRA; results in the research Claude Doc's Log)** — what it means for training: **tag-style prompts beat descriptive sentences** (+0.67 on a 1–5 scale) → train and demo with key-value tags; **120 s renders beat 30 s on every prompt** (+1.0) → render demos at 120 s, or checkpoints get judged at the model's weakest length; **CFG 3 and the base:ARC 1:2 merge gave nothing** → ARC demos stay CFG 1 / 8 steps, never train against the merge; **squeakfix** (decoder LoRA) is neutral on fresh renders and only helps across encode/decode round trips → irrelevant during training, use it for audio-to-audio at inference. Rating noise was ±1 point and Audiobox/CLAP barely separated variants → choose checkpoints by ear over several prompts and seeds.
- **Data prep.** Clean beats more: drop off-style tracks, trim long silent intros/outros (Pinokio). Don't transcode MP3 → WAV (adds nothing). Loudness normalisation: one experimental source only — skip.

## Running a job (agent procedure)

Payload JSONs live in `AI_Audio/sa3lab/train/` (`djrum_uts_v1.json` is the template: layer filter, prompt config, demo set). Everything long-lived is detached so the agent's shell can die without killing anything.

1. **Preflight:** `pgrep -af "python[^ ]* .*[l]ora_train.py"` empty; `nvidia-smi` ~0 %; `python3 ../sa3lab/train/uf.py runs` answers (otherwise start the dashboard as in Setup); `df -h ~/SSD2TB` (a 14k run adds ~0.6 GB).
2. **Launch:** `python3 ../sa3lab/train/uf.py launch <payload.json>` → run id `<name>-<timestamp>`.
3. **GPU log** (the UI keeps only 1 h of VRAM): `setsid nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv -l 60 > state/runs/<run-id>.gpu.csv 2>&1 < /dev/null & disown`. **Stop it by PID**, not `pkill -f "nvidia-smi --query-gpu"`: the dashboard runs its own `nvidia-smi --query-gpu=… -lms 500` poller and that pattern kills it too.
4. **Sanity within ~3 min** in `state/runs/<run-id>.log`: `lora layers: 168`, `Trainable LoRA params: 19,058,688`, one `Generating demo` per demo entry (step-0 baseline), then `Step N` lines. 228/229 layers = the payload lost `lora_exclude` → `uf.py kill <run-id>`.
5. **Watch:** Claude's Monitor tool on `tail -n 0 -F <log> | tr '\r' '\n' | grep -E --line-buffered "Step [0-9]*000,|Training done|Traceback|Error|out of memory|Killed"` plus a 60 s `pgrep -f "[l]ora_train.py"` liveness loop (30 min per arm; re-arm). Don't append `awk` to the pipe: Ubuntu's `mawk` block-buffers its input and the watch goes silent. Each `Step N000` appears 2–3 times (tqdm refreshes); the step after a demo round shows ~25 s/it, which is the demo pause, not a stall. A real stall = no step progress for >10 min outside a demo round.
6. **Every checkpoint:** wait until the step is ~15 past it (demo round done), then `uf.py collect` (copies demos to `AI_Audio/outputs/underfit-<name>/` as `sa3med_<name>_step<N>_d<i>_<sampler>_<prompt-slug>_s<seed>.mp3`; the slug strips the shared prefix, so a prefix-only prompt is labelled `uncond`, and the `d<i>` index is what tells rows apart).
7. **Crash:** read `<run-id>.log`, `.log.exit` (traceback), `.log.bash.err`; fix environmental causes; `uf.py resume <run-id>` (untested; dashboard restart control as fallback). Never relaunch from scratch under the same name. Same failure twice → stop and ask Xander.
8. **Wrap up:** on `Training done`, stop the GPU logger, `uf.py collect`, report a step × demo grid of `file://` links. The agent doesn't judge sound; Xander listens.

**Our standard demo set** (12 × 120 s, one fixed seed per row so a row is an A/B across steps; step 0 = no-LoRA baseline; all but d1 start with the training prefix):

| # | Sampler | Prompt | Seed | Tests |
|---|---|---|---|---|
| d0 | ARC 8 steps, CFG 1 | the exact training prompt | 42 | the learned style, rendered the way we render at inference — the main one |
| d1 | ARC | empty | 44 | style absorbed unconditionally (a late-run sign) |
| d2, d3 | ARC | dataset-adjacent tag prompts, no trigger (Djrum: breakbeat/jungle 170 BPM; ambient/modern classical) | 101, 102 | does the LoRA pull in-style prompts toward the dataset? |
| d4–d9 | ARC | jazz trio · cinematic orchestral · techno · folk · boom-bap · metal | 201–206 | collapse check: each should stay its genre; the first step where it flips toward the dataset marks the last checkpoint that still generalises |
| d10 | RF base 50 steps, CFG 7 | training prompt | 42 | lights up first when the run "gets it" |
| d11 | RF base, CFG 1 | training prompt | 42 | should overtake d10 once CFG 7 over-cooks |

## Choosing a checkpoint and using it

- The **loss-curve elbow** (where loss first starts dropping) is often the most interesting checkpoint; past it the run creeps toward memorisation.
- Signs a run has got it: CFG 7 RF demos sound right first; later CFG 1 sounds cleaner than CFG 7; then even empty-prompt demos take on the style. ARC demos lag but end up cleaner.
- Keep several checkpoints: memorised ones are still good for audio-to-audio style transfer. The last one is often not the best.
- **Reading the dashboard charts** (from `djrum-uts-v1`, 14k steps on ~80 windows):
  - There is no held-out set, so no chart can prove overfitting; the demos are the test. The most telling chart is **Loss by Noise Level**. A steady drop at high noise (σ 0.6–1.0: 0.92 → 0.80 there) means the model is getting better at guessing a whole passage from almost pure noise, i.e. learning (and eventually memorising) the dataset's specific material. Low-noise buckets (fine texture) stayed flat.
  - The σ < 0.4 lines are sparse: the sampler puts only ~1% of steps below σ 0.2. Their EMA wanders (it even "rises") on noise alone; average per 2k-step window from `demos/loss_by_timestep.bin` (`struct "Iff"`: step, σ, loss) before reading anything into them.
  - **LoRA Magnitude** is the global L2 norm of every trainable tensor. For DoRA that's dominated by the magnitude vectors, which start at the base weights' row norms (~2,290 for 168 layers). A slow ~1.3 % decline over 13k steps is just AdamW `weight_decay: 0.01` (from `training_template.json`) at LR 1e-4, i.e. (1 − 1e-6)^13,000. It is not a learning signal; a fast rise or fall would be.
  - The 200-step average loss is too noisy to call a plateau; 2k-window means kept falling through 14k (0.82 → 0.72).
- Inference: LoRA **strength 0.6–0.8** gives "in the style of" without regurgitation; skipping the LoRA on the first denoising step lets the base model set the song structure. Base-trained LoRAs work on ARC. Resuming from a LoRA on a second dataset fuses styles (dadabots: 10k steps at 47 s, then 300 at 12 s on another style).
- Data rules of thumb: 10 min floor, 30+ better, **one style per dataset** (several styles → one LoRA each, stacked at inference). Latents are codec-locked: medium ↔ SAME-L, small ↔ SAME-S.

## Our fork's changes to dashboard defaults

`dashboard/index.html` `openNewFinetuneModal` / `_updateCropLength` (each tagged `fork:` in a comment): crop **512 latents** on non-Apple (upstream: model max, 4096 ≈ 380 s), steps **10k** (upstream 40k), checkpoint every **500** (upstream 1k), base precision **fp32** on non-Apple (upstream fp16). Keep these edits tiny — every `git merge upstream/main` has to carry them.

## Known issues

- **JSON/txt sidecars in a `json/` or `txt/` subfolder are never found** (upstream issue #5): `pre_encode.py` looks in `fp.parent.parent / "json"`, one level too high. Put sidecars next to the audio.
- **MLX engine drops the prompt config** (Mac only): `underfit/backends/mlx_engine.py` `build_trainer_cmd` never passes `--prompt-config`, so trigger/fixed/path prompts are silently ignored and training falls back to tag prompts or unconditional. Fix if we ever train via MLX: write `dataset_config["prompt_config"]` to a temp JSON and pass it.
- **Unverified:** the MLX pre-encoder (`stable-audio-3/optimized/mlx/scripts/pre_encode_mlx.py`) claims to write this exact latent format; untested against the torch trainer. Encode on the GPU box.
- Upstream's `uv.lock` pinned torch `+cu126` while `pyproject.toml` asks for the cu128 index, so the first `uv sync` re-locks to `+cu128` (what we run). The re-lock is correct; committing it is fine but adds churn to upstream merges.
