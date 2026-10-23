# AGENTS.md — reproducing PRISM

Instructions for AI coding agents asked to reproduce results from this repository
(paper: arXiv 2606.07974). Humans: see `README.md`; the commands are the same.

## Rules

- Do not modify method code, configs, seeds or hyperparameters to make numbers match. If a step
  fails, stop and report the command, the error message and the log path. Fixing the Python environment
  inside the project venv (installing or re-pinning a package) is allowed; report exactly what you changed.
- Ask the user before (a) downloading a dataset (sizes in §1), (b) installing system packages
  (`sudo`, `apt`, drivers, CUDA), (c) launching the full Fig. 5 sweep (§6, about 6 GPU-hours) or
  world-model training (§8).
- Run every command from the repository root, with the same `STABLEWM_HOME` in every shell.
- Report results as a table: `task | planner | K | seed | measured success rate | expected success rate`,
  followed by the log paths of any mismatching cell.

## 1. Requirements

- Linux, one NVIDIA GPU (tested: RTX 5090 32 GB, CUDA 12.8), Python 3.10, `uv` or `pip`, `zstd`.
- Rendering is headless: the evaluation scripts set `MUJOCO_GL=egl`, so EGL must be available.
- Disk per task. `download.sh` keeps the compressed archive after expanding it; it can be deleted
  afterwards.

| Task | Archive | Expanded dataset | Peak during download | Feature cache |
|---|---|---|---|---|
| TwoRoom | 3.4 GB | 12.8 GB | 16 GB | 0.3 GB |
| PushT | 13 GB | 46 GB | 59 GB | 0.9 GB |
| Reacher | 24 GB | 99 GB | 123 GB | 0.7 GB |
| Cube | 46 GB | 102 GB | 148 GB | 0.7 GB |

TwoRoom is the cheapest task for a first check.

## 2. Setup

```bash
nvidia-smi                                  # note "CUDA Version" (the newest CUDA the driver supports)
uv venv --python 3.10 && source .venv/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
export STABLEWM_HOME=$HOME/.stable-wm      # datasets, world models, feature caches, rollout videos
python -c "import torch; torch.zeros(1).cuda(); print(torch.__version__, torch.cuda.get_device_name(0))"
python -c "import stable_worldmodel, stable_pretraining, imageio_ffmpeg; print('ok')"
zstd --version && hf version             # needed by scripts/download.sh
```

- Install torch from the PyTorch index before `requirements.txt`: the default PyPI wheel can target a
  newer CUDA than the driver supports. `cu128` needs a driver reporting CUDA ≥ 12.8; otherwise use the
  `cu126` or `cu124` index with the same torch/torchvision versions.
- The CUDA check must allocate a tensor. `torch.cuda.is_available()` alone can print `True` on an
  incompatible driver once Lightning is imported (it switches the check to an NVML device count).
- `imageio-ffmpeg` is required: evaluation writes rollout videos, and without it imageio falls back to
  its TIFF writer and the run fails with `TiffWriter.write() got an unexpected keyword argument 'fps'`.

- Do not install `stable-worldmodel[env]`: it pulls `stable-baselines3` → `gym==0.21`, which does not
  build with current setuptools. `requirements.txt` lists the environment packages directly.
- Keep `stable-worldmodel==0.0.6` and `stable-pretraining==0.1.6`; newer versions changed the
  dataset and solver APIs this code uses.
- If the CUDA check fails with "NVIDIA driver on your system is too old", reinstall torch/torchvision from
  the index matching `nvidia-smi`'s CUDA version. Ask the user before touching drivers.

## 3. Data and weights

```bash
bash scripts/download.sh pusht            # pusht | cube | tworoom | reacher | all
```

Afterwards these files must exist:

| Task | `$STABLEWM_HOME/…` dataset | `$STABLEWM_HOME/…` world model | prior head |
|---|---|---|---|
| PushT | `pusht_expert_train.h5` | `pusht/lewm_object.ckpt` | `checkpoints/prior_head_pusht.pt` |
| Cube | `ogbench/cube_single_expert.h5` | `cube/lewm_object.ckpt` | `checkpoints/prior_head_cube.pt` |
| TwoRoom | `tworoom.h5` | `tworooms/lewm_object.ckpt` | `checkpoints/prior_head_tworoom.pt` |
| Reacher | `dmc/reacher_random.h5` | `reacher/lewm_object.ckpt` | `checkpoints/prior_head_reacher.pt` |

The Reacher archive contains `reacher.h5`; `download.sh` renames it to `dmc/reacher_random.h5`.

## 4. Quick check (about 2-5 minutes per task after the download)

```bash
bash scripts/test.sh pusht --K 32 --seeds 0
```

This runs vanilla, warm-start and PRISM MPPI once each (50 episodes per run). Expected success rates
(%) for seed 0, K = 32:

| Task | vanilla | warm_start | prism |
|---|---|---|---|
| PushT | 60 | 76 | 84 |
| Cube | 48 | 54 | 82 |
| TwoRoom | 68 | 82 | 82 |
| Reacher | 44 | 48 | 60 |

Most cells reproduce exactly. A cell can differ by one or two episodes (2-4 pp) between runs because
of nondeterministic GPU kernels; PushT warm-start has given 74, 76 and 78 in repeated runs of identical
code. A gap larger than 4 pp in any cell indicates a setup problem: check the data and weight paths,
`STABLEWM_HOME`, and the package versions before anything else.

Mean-only execution (the prior's mean without sampling, paper Table II), seed 0:
`bash scripts/test.sh <task> --K 32 --seeds 0 --methods mean_only` → PushT 32, Cube 66, TwoRoom 74, Reacher 6.

Each run appends a row to `results/<task>.tsv` and writes its log to `logs/`. The summary printed at the
end averages every row in that file, so delete `results/<task>.tsv` before a fresh reproduction.

## 5. Seed 0 at all budgets (optional, about 10 minutes per task)

```bash
bash scripts/test.sh pusht --seeds 0
```

| Task | vanilla K=32/64/128 | warm_start K=32/64/128 | prism K=32/64/128 |
|---|---|---|---|
| PushT | 60 / 58 / 64 | 76 / 74 / 66 | 84 / 82 / 92 |
| Cube | 48 / 50 / 48 | 54 / 54 / 56 | 82 / 82 / 86 |
| TwoRoom | 68 / 66 / 80 | 82 / 72 / 72 | 82 / 86 / 86 |
| Reacher | 44 / 46 / 50 | 48 / 42 / 46 | 60 / 54 / 64 |

Same tolerance as §4.

## 6. Full paper Fig. 5 (ask first: 360 runs, about 6 GPU-hours on an RTX 5090)

```bash
bash scripts/test.sh all --seeds "0 1 2 3 4 5 6 7 8 9"
```

Expected means over the ten seeds (%); the README lists the standard deviations. Each mean should
land within about 2 pp.

| Task | vanilla K=32/64/128 | warm_start K=32/64/128 | prism K=32/64/128 |
|---|---|---|---|
| PushT | 56.0 / 58.6 / 58.8 | 70.6 / 71.0 / 68.8 | 80.6 / 82.8 / 85.4 |
| Cube | 47.4 / 48.0 / 48.2 | 56.6 / 57.2 / 58.8 | 82.8 / 83.0 / 83.0 |
| TwoRoom | 66.0 / 69.4 / 73.4 | 74.8 / 78.2 / 75.8 | 81.8 / 85.8 / 86.4 |
| Reacher | 48.2 / 48.2 / 43.8 | 46.2 / 48.4 / 46.8 | 60.0 / 60.4 / 64.4 |

## 7. Training the prior head

```bash
cp checkpoints/prior_head_pusht.pt checkpoints/prior_head_pusht.released.pt   # train.sh overwrites the downloaded head
bash scripts/train.sh pusht
```

This caches frozen encoder features (`$STABLEWM_HOME/jepa_cache/<task>/`; TwoRoom 4 min, the other
tasks 9-11 min) and trains the head (2-5 min). The last lines of the output report the best checkpoint:

| Task | best val NLL | best val MSE | best epoch |
|---|---|---|---|
| PushT | 0.5179 | 0.7331 | 1 |
| Cube | 0.1920 | 0.6077 | 7 |
| TwoRoom | 0.4442 | 0.9281 | 1 |
| Reacher | 0.4950 | 0.9938 | 1 |

PushT trains for 15 epochs (the released head's schedule), the other tasks for 50. With the same package
versions on an RTX 5090, all four tasks retrain to weights identical to the released ones (compare the `state_dict` tensors; the files' hashes differ because checkpoints store
per-epoch timings, and the released files carry a few extra `args` keys from an earlier version of the
training script; evaluation reads only `config` and `state_dict`). On other hardware expect differences
in the third or fourth decimal. To compare a retrained head with the released one:

```bash
python -c "import torch,sys; a,b=(torch.load(f,map_location='cpu',weights_only=False)['state_dict'] for f in sys.argv[1:]); print(max((a[k]-b[k]).abs().max().item() for k in a))" \
    checkpoints/prior_head_pusht.released.pt checkpoints/prior_head_pusht.pt     # 0.0 = identical
``` A best epoch
of 1 is expected for PushT, TwoRoom and Reacher: validation NLL rises after the first epoch and the
lowest-NLL checkpoint is kept. TwoRoom and Reacher print a note that val MSE fell by less than 15%;
this is expected (paper Table I). After training, `bash scripts/test.sh <task> --K 32 --seeds 0`
evaluates the new head against the §4 table.

## 8. World-model training (optional, many GPU-hours; ask first)

```bash
bash scripts/train.sh pusht --world-model
```

This trains LeWM from scratch for 100 epochs and replaces `$STABLEWM_HOME/<wm dir>/lewm_object.ckpt`
(the previous file is kept as `lewm_object.ckpt.bak`). The paper uses the released upstream LeWM
weights, so no reference numbers exist for retrained world models; downstream success rates will
differ from §4-§6.

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| `run failed — see logs/...` | the named log; `test.sh` prints its last 15 lines, the traceback is at the end |
| `TiffWriter.write() got an unexpected keyword argument 'fps'` | `imageio-ffmpeg` is missing (§2) |
| `NVIDIA driver on your system is too old` | torch was installed for a newer CUDA than the driver supports (§2) |
| EGL / OpenGL error at environment creation | headless EGL libraries (`libegl1`); ask the user before installing system packages |
| `FileNotFoundError` for an `.h5` or `.ckpt` | `STABLEWM_HOME` differs between the download shell and the current shell, or `download.sh` did not finish |
| success rates near 0 for every planner | wrong world model or dataset for the task (§3 table) |
| `ModuleNotFoundError: gym` / build errors for `gym==0.21` | `stable-worldmodel[env]` was installed; use `requirements.txt` instead |
| `ImportError: cannot import name 'config' from 'datasets'` | an old `datasets` was resolved; rerun `uv pip install -r requirements.txt` (it pins `datasets` and `fsspec`) |

Harmless output: gymnasium warnings ("Casting input x to numpy array", "obs ... not within the observation
space") in every evaluation log, and uv's `typer ... does not have an extra named all` during install.

Outputs: `results/<task>.tsv` (one row per run), `logs/` (one log per run), `outputs/` (Hydra run
directories), rollout videos under `$STABLEWM_HOME/<wm dir>/`.
