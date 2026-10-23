# PRISM: PRior-guided Imagination Sampling in world Models

[Paper](https://arxiv.org/abs/2606.07974) · [Project page](https://yuhaiw.github.io/PRISM_web/) · Weights:
[PushT](https://huggingface.co/YuhaiW/prism-jepa-pusht) ·
[Cube](https://huggingface.co/YuhaiW/prism-jepa-cube) ·
[TwoRoom](https://huggingface.co/YuhaiW/prism-jepa-tworoom) ·
[Reacher](https://huggingface.co/YuhaiW/prism-jepa-reacher)

PRISM trains a small MLP head on the frozen encoder of a JEPA world model
([LeWM](https://github.com/lucas-maes/le-wm)) to predict a goal-conditioned Gaussian action prior
(μ_p, σ_p). At every replanning step the prior is fused with the planner's default proposal
N(0, σ_π²) by a closed-form Product of Gaussians, and MPPI samples from the fused distribution while
the world model scores candidates by embedding distance to the goal.

This repository contains training and evaluation for the four simulated tasks of the paper:
PushT, Cube (OGBench `cube-single`), TwoRoom and Reacher.

## Reproducing with an AI agent

`AGENTS.md` gives coding agents (Claude Code, Codex, Cursor, …) step-by-step instructions with the
expected numbers for every check. Paste this prompt into your agent, changing the task if needed:

```text
Reproduce the PRISM results (arXiv 2606.07974) for the TwoRoom task on this machine.
1. Clone https://github.com/YuhaiW/prism-jepa and follow AGENTS.md in the repo root.
2. Before downloading datasets (3-46 GB each) or installing system packages, tell me what and how large, and wait for my OK.
3. Run the quick check first and report a table: task, planner, K, seed, your success rate, expected success rate.
4. Do not change configs, seeds or hyperparameters. If a step fails, report the error and the log path instead of modifying the method code.
```

For the full paper Fig. 5, replace the first line with "Reproduce Fig. 5 of the PRISM paper
(arXiv 2606.07974) for all four tasks on this machine." and expect about 6 GPU-hours.

<p align="center">
  <img src="https://yuhaiw.github.io/PRISM_web/static/images/fig_architecture.png" width="90%" alt="PRISM architecture">
</p>

## Results

Success rate (%), mean ± std over ten seeds {0, …, 9}, 50 episodes per seed, K candidates per MPPI
iteration (paper Fig. 5). All planners: goal 25 environment steps ahead, H = 5, B = 5, J = 30,
λ = 0.5, K_elite = 30, prior scale s = 1.

| Task | Planner | K = 32 | K = 64 | K = 128 |
|---|---|---|---|---|
| PushT | Vanilla MPPI | 56.0 ± 5.7 | 58.6 ± 6.2 | 58.8 ± 6.9 |
| | Warm-start MPPI | 70.6 ± 9.1 | 71.0 ± 5.9 | 68.8 ± 6.8 |
| | **PRISM-MPPI** | **80.6 ± 5.6** | **82.8 ± 6.9** | **85.4 ± 7.1** |
| Cube | Vanilla MPPI | 47.4 ± 6.3 | 48.0 ± 8.3 | 48.2 ± 7.2 |
| | Warm-start MPPI | 56.6 ± 10.1 | 57.2 ± 8.0 | 58.8 ± 9.4 |
| | **PRISM-MPPI** | **82.8 ± 5.5** | **83.0 ± 6.1** | **83.0 ± 4.7** |
| TwoRoom | Vanilla MPPI | 66.0 ± 5.7 | 69.4 ± 7.2 | 73.4 ± 5.7 |
| | Warm-start MPPI | 74.8 ± 6.3 | 78.2 ± 6.6 | 75.8 ± 3.6 |
| | **PRISM-MPPI** | **81.8 ± 5.8** | **85.8 ± 4.5** | **86.4 ± 4.2** |
| Reacher | Vanilla MPPI | 48.2 ± 7.0 | 48.2 ± 4.0 | 43.8 ± 8.1 |
| | Warm-start MPPI | 46.2 ± 6.4 | 48.4 ± 4.5 | 46.8 ± 7.8 |
| | **PRISM-MPPI** | **60.0 ± 6.9** | **60.4 ± 5.6** | **64.4 ± 5.0** |

Vanilla MPPI starts from N(0, σ_π²); warm-start MPPI starts from μ_p with width σ_π; PRISM-MPPI starts
from the PoG-fused (μ_p, σ_p).

## Installation

Tested with Python 3.10, PyTorch 2.8 (CUDA 12.8) and one RTX 5090.

```bash
git clone https://github.com/YuhaiW/prism-jepa.git && cd prism-jepa
uv venv --python 3.10 && source .venv/bin/activate     # or any Python 3.10 environment
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
python -c "import torch; torch.zeros(1).cuda(); print(torch.__version__, torch.cuda.get_device_name(0))"
export STABLEWM_HOME=$HOME/.stable-wm                   # datasets, world models and feature caches live here
```

`cu128` matches drivers that support CUDA 12.8 (`nvidia-smi` shows the supported version); use the
`cu126`/`cu124` index for older drivers.

`scripts/download.sh` also needs the `hf` CLI (installed with `huggingface_hub`) and `zstd`.

## Data and pretrained weights

```bash
bash scripts/download.sh cube        # or pusht | tworoom | reacher | all
```

For each task this downloads

- the dataset from [`quentinll/lewm-<task>`](https://huggingface.co/collections/quentinll/lewm-69c58f5e20580628114d20c4)
  into `$STABLEWM_HOME` (compressed: PushT 13 GB, Cube 46 GB, TwoRoom 3.4 GB, Reacher 24 GB; each expands 3-4×),
- the LeWM world model and the PRISM prior head from `YuhaiW/prism-jepa-<task>`
  into `$STABLEWM_HOME/<task>/lewm_object.ckpt` and `checkpoints/prior_head_<task>.pt`.

`--no-data` or `--no-weights` skips either part. The world models are the upstream LeWM weights; to
build them from [`quentinll/lewm-<task>`](https://huggingface.co/quentinll) instead, download that model
repo to `$STABLEWM_HOME/hf_<task>/` and run `python convert_hf_checkpoint.py <task>`.

| Task | Dataset (`$STABLEWM_HOME/…`) | World model (`$STABLEWM_HOME/…`) |
|---|---|---|
| PushT | `pusht_expert_train.h5` | `pusht/lewm_object.ckpt` |
| Cube | `ogbench/cube_single_expert.h5` | `cube/lewm_object.ckpt` |
| TwoRoom | `tworoom.h5` | `tworooms/lewm_object.ckpt` |
| Reacher | `dmc/reacher_random.h5` | `reacher/lewm_object.ckpt` |

## Training

```bash
bash scripts/train.sh cube                    # feature cache + prior head
bash scripts/train.sh cube --world-model      # also train the LeWM world model from scratch first
```

1. `--world-model` only: `train.py` trains LeWM for 100 epochs (`config/train/`) and copies the last
   epoch to `$STABLEWM_HOME/<task>/lewm_object.ckpt`.
2. `jepa_cache.py` encodes every frame of the dataset with the frozen encoder
   (`$STABLEWM_HOME/jepa_cache/<task>/`, 4-11 minutes per task).
3. `train_prior_head.py` trains the head with β-NLL (β = 0.5) for 50 epochs on a 90/10 episode split
   and writes `checkpoints/prior_head_<task>.pt` (2-5 minutes per task).

Extra arguments after the task are passed to `train_prior_head.py` (`--epochs`, `--lr`, `--seed`, …).

## Evaluation

```bash
bash scripts/test.sh cube                                       # K = 32, 64, 128; seed 0
bash scripts/test.sh cube --seeds "0 1 2 3 4 5 6 7 8 9"         # the paper's Fig. 5
bash scripts/test.sh all --K 128 --methods "vanilla prism"
bash scripts/test.sh cube --solver cem                          # the same prior under CEM
```

Methods: `vanilla`, `warm_start`, `prism`, and `mean_only` (execute μ_p without sampling, paper
Table II). Each run appends one row to `results/<task>.tsv` and logs to `logs/`; the script ends with a
mean ± std summary over all rows in that file. A single run is

```bash
python eval_prism_head.py --config-name=cube policy=cube/lewm solver=mppi solver.num_samples=128 seed=0 \
    +head.ckpt=checkpoints/prior_head_cube.pt +head.injection_mode=pog +head.sigma_scale=1.0
python eval.py --config-name=cube policy=cube/lewm solver=mppi solver.num_samples=128 seed=0
```

## Repository layout

```
prism_mppi.py          MPPI with PoG fusion of an external prior (PrismMPPISolver)
prism_cem.py           the same for CEM (PrismCEMSolver)
prism_head_policy.py   policy that runs the prior head and hands (μ_p, σ_p) to the solver
prior_head.py          prior head (2 × 512 GELU MLP) and β-NLL loss
jepa_cache.py          frozen-encoder feature cache for head training
train_prior_head.py    prior-head training
eval_prism_head.py     evaluation with a prior (prism / warm_start / mean_only)
eval.py                evaluation without a prior (vanilla)
train.py               LeWM world-model training
jepa.py, module.py, utils.py, convert_hf_checkpoint.py    LeWM model code (upstream)
config/                Hydra configs: eval/ per task and solver, train/ per dataset
scripts/               download.sh, train.sh, test.sh (common.sh holds the per-task settings)
AGENTS.md              reproduction instructions and expected numbers for AI agents (CLAUDE.md imports it)
```

## Citation

```bibtex
@misc{wang2026prism,
  title         = {PRISM: PRior-guided Imagination Sampling in world Models},
  author        = {Yuhai Wang and Jiawei Xia and Bowen Sun and Rongxuan Zhou and Xiao Hu and Yongliang Shi and Jing Du and Yang Ye},
  year          = {2026},
  eprint        = {2606.07974},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.07974},
}
```

## Acknowledgements

The world model, datasets and environments come from
[LeWM](https://github.com/lucas-maes/le-wm) and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). MIT License (see `LICENSE`).
