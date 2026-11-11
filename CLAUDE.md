# underfit — our fork

## Mission

Xander's fork (`aiXander/underfit`, a submodule of `AI_Audio`) of dadabots' `dada-bots/underfit`: a web dashboard + PyTorch training loop for LoRA/DoRA finetunes of Stable Audio 3. We use it to train SA3-medium adapters on our own audio on the GPU box (rednax-beast, RTX 3090 24 GB); the resulting `.safetensors` get rendered on the M2 Mac via `AI_Audio/sa3lab`. Nothing is deployed. It needs `../stable-audio-3` (our SA3 fork) as a sibling — it's installed editable into this repo's `.venv`.

## Blast radius

- **The fork is public.** Never commit `state/` (runs, datasets, checkpoints, model symlinks — gitignored), audio, weights or tokens. `.gitignore` ignores `CLAUDE.md`: commit it with `git add -f`. `AGENTS.md` is a symlink to it.
- **Never push to `upstream` (dadabots).** Commit on `main`, push `origin`, then bump the pointer in `AI_Audio` (order and details: `AI_Audio/CLAUDE.md`, "Git").
- **The dashboard binds `0.0.0.0` by default** — always start it with `UNDERFIT_DASHBOARD_HOST=127.0.0.1`.
- **Downloads go to `AI_Audio/models/hf-cache`** (`HF_HUB_CACHE`), never `~/.cache`. Ask before pulling new model packs (each is 7–25 GB).
- **Free to run without asking:** the dashboard, dataset encodes, training runs and demos on the local GPU, `uv sync`, reading HF/GitHub pages.

## Documentation map

| Doc | Open when |
|---|---|
| [docs/reference/sa3-lora-training.md](docs/reference/sa3-lora-training.md) | setting up or debugging training on the GPU box, **launching and babysitting a run** (agent procedure, standard 12-demo set), reading the dashboard charts, scripting the dashboard API, choosing knobs (steps, crop, prompts, layer filter) and the evidence behind them, picking checkpoints, or hitting a known underfit bug |
| [docs/TODO/](docs/TODO/) | continuing planned work — `sa3-lora-finetune.md` holds what's left of the Djrum v1 run (trained 2026-11-05; listening, checkpoint shortlist, Mac renders, v2 ideas) |
| [README.md](README.md) | upstream's dashboard manual (UI walkthrough, adapter formats, Colab). Leave it as upstream wrote it |

## Standing debugging priors

- Download "stuck" at a few MB in the cache dir → xet buffers in RAM first; check NIC throughput / process RSS before killing it.
- `401` / `GatedRepoError` on ARC → `HF_TOKEN` not exported, or the SA3 licence not accepted for that account.
- A scripted run trains far more layers than a UI run → the API has no default `lora_exclude`; the UI's default layer filter lives in `dashboard/index.html` `_applyDefaultExclusions`.

## Repo layout

- `dashboard/` — `server.py` (HTTP API + process manager, ~8k lines; runs are detached subprocesses tracked in `state/runs.json`), `index.html` (the whole UI; form defaults in `openNewFinetuneModal`), `models/<pack>/registry.json` (per-model paths, VRAM numbers, layer-filter chips).
- `underfit/` — the package: `training/loop.py` (training loop), `backends/sa3.py` (dataset + model glue for SA3), `backends/mlx_engine.py` (Mac path, has a prompt-config bug), `cli/setup.py` (install wizard + model download).
- `dataset_processing/` — `pre_encode.py` (audio → latents; 600 s cap), `prompt_templates.py` (per-step prompt builder).
- `lora_train.py`, `run_gradio.py` — trainer and inference entry points the dashboard shells out to. `defaults.ini` fills flags the dashboard doesn't pass.
- `state/` — all runtime data (gitignored): `models/` (symlinks into `AI_Audio/models/hf-cache`), `datasets/`, `runs/<run>/<hash>/checkpoints/*.safetensors`, `runs/<run>/demos/`.

## Load-bearing rules

1. Keep edits to upstream files tiny and tagged with a `fork:` comment; every `git merge upstream/main` must carry them. Current ones: the **+ Finetune** form defaults (reference doc, "Our fork's changes").
2. Source audio stays in `AI_Audio/audio-data/<artist>/<release>/`; copy demos you want Xander to hear into `AI_Audio/outputs/` with descriptive names.
3. Sidecar captions go next to the audio file, not in a `json/`/`txt/` subfolder (upstream bug, see reference doc).

## Definition of done

- A training change was proven by a short run (≥100 steps, demos render, checkpoint validates), not just by reading code.
- Findings about the pipeline or knobs are in `docs/reference/sa3-lora-training.md`; plans in `docs/TODO/`; sound results and settings in the SA3 research Claude Doc (URL in `AI_Audio/CLAUDE.md`).
- Your summary names which repos (this fork, `stable-audio-3`, `AI_Audio`) hold uncommitted or unpushed work.

## Commands

- Install / update: `uv sync --inexact`, then `HF_HUB_CACHE=<AI_Audio>/models/hf-cache .venv/bin/python -m underfit.cli.setup --backend sa3 --backend-path ../stable-audio-3 --models sa3-medium` (needs `HF_TOKEN`). Plain `uv sync` (no `--inexact`) uninstalls the editable SA3 backend.
- Dashboard: `setsid nohup env UNDERFIT_DASHBOARD_HOST=127.0.0.1 HF_HUB_CACHE=<AI_Audio>/models/hf-cache ./run.sh > state/dashboard.log 2>&1 < /dev/null & disown` → http://127.0.0.1:8787 (from the Mac: `ssh -L 8787:127.0.0.1:8787 <gpu-box>`). Detached on purpose: trainers it spawns run in their own session (`os.setsid`), so neither dies with an agent's shell. Check for a running trainer with `pgrep -af "[l]ora_train.py"` (the bracket stops pgrep matching itself).
- Scripted use: `python3 ../sa3lab/train/uf.py launch <payload.json> | status [run] | runs | collect [run]` — launches through the same API as the UI from a payload JSON (ours: `../sa3lab/train/*.json`), reports progress, copies demos to `AI_Audio/outputs/` with descriptive names. Raw API: `POST /api/datasets/encode` `{name, input_dir, model, gpus:[0]}`, `POST /api/runs/new` (payload shape: `confirmNewFinetune` in `index.html`; pass `lora_exclude` yourself).
- Diagnostics: `.venv/bin/python -c "from underfit.monitor import debug_info; debug_info()"`.
