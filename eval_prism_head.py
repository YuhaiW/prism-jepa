"""eval_prism_head.py — evaluate a planner guided by a trained PriorHead.

Uses the same StandardScaler / transform / world setup as eval.py so the head's
normalized actions decode through the same inverse_transform path.

Usage:
    # PRISM-MPPI (paper setting: s = 1, K_elite = 30 from config/eval/solver/mppi.yaml)
    python eval_prism_head.py --config-name=cube policy=cube/lewm solver=mppi \\
        solver.num_samples=128 +head.ckpt=checkpoints/prior_head_cube.pt \\
        +head.injection_mode=pog +head.sigma_scale=1.0

    # warm-start MPPI (prior mean, planner width) and mean-only execution
    ...  +head.injection_mode=warm_start
    ...  +head.injection_mode=bc_only

    # the same prior under CEM
    python eval_prism_head.py --config-name=cube policy=cube/lewm solver=cem ...

head.* options:
    ckpt            prior-head checkpoint (default: checkpoints/prior_head_<config-name>.pt)
    injection_mode  pog | warm_start | none | bc_only   (default: pog)
    sigma_scale     prior scale s                        (default: 1.0)
    jepa_object     world-model object ckpt (default: $STABLEWM_HOME/<policy dir>/lewm_object.ckpt)
"""
import os

os.environ["MUJOCO_GL"] = "egl"

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from prism_mppi import PrismMPPISolver  # noqa: E402
from prism_cem import PrismCEMSolver  # noqa: E402
from prism_head_policy import PrismHeadPolicy  # noqa: E402


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name, keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


@hydra.main(version_base=None, config_path="./config/eval", config_name="cube")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"
    task = HydraConfig.get().job.config_name  # pusht | cube | tworoom | reacher

    # ---- Head config ----
    head_cfg = cfg.get("head", {})
    injection_mode = head_cfg.get("injection_mode", "pog")
    sigma_scale = head_cfg.get("sigma_scale", 1.0)
    head_ckpt = head_cfg.get("ckpt", str(REPO / "checkpoints" / f"prior_head_{task}.pt"))
    # Default jepa object path follows the LeWM convention <STABLEWM_HOME>/<task>/lewm_object.ckpt.
    default_jepa = (
        Path(swm.data.utils.get_cache_dir())
        / cfg.policy.replace("/lewm", "")
        / "lewm_object.ckpt"
    )
    jepa_object = head_cfg.get("jepa_object", str(default_jepa))

    print("\n" + "=" * 64)
    print(f"  PrismHead eval — INJECTION_MODE = {injection_mode!r}")
    print(f"  sigma_scale    = {sigma_scale}")
    print(f"  head_ckpt      = {head_ckpt}")
    print(f"  jepa_object    = {jepa_object}")
    print(f"  num_eval       = {cfg.eval.num_eval}")
    print(f"  policy         = {cfg.policy}")
    print(f"  dataset        = {cfg.eval.dataset_name}")
    print(f"  seed           = {cfg.seed}")
    print("=" * 64 + "\n")

    # ---- Build env / world ----
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    # StandardScaler over every non-pixel column, exactly as in eval.py.
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ("pixels",):
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # ---- Build model + PRISM solver + PrismHeadPolicy ----
    assert cfg.policy != "random", "PRISM eval requires a world-model checkpoint (policy=<task>/lewm)"
    model = swm.policy.AutoCostModel(cfg.policy).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver_target = cfg.solver.get("_target_", "stable_worldmodel.solver.MPPISolver")
    solver_kwargs = OmegaConf.to_container(cfg.solver, resolve=True)
    solver_kwargs.pop("_target_", None)
    solver_kwargs.pop("model", None)
    # injection_mode and sigma_scale are owned by the policy, which forwards them to the solver.
    if "MPPI" in solver_target:
        solver = PrismMPPISolver(model=model, **solver_kwargs)
    elif "CEM" in solver_target:
        solver_kwargs.pop("temperature", None)  # CEMSolver has no temperature
        solver = PrismCEMSolver(model=model, **solver_kwargs)
    else:
        raise ValueError(
            f"Unsupported solver target: {solver_target!r}. Expected an MPPI or CEM variant."
        )
    print(f"[eval_prism_head] solver class: {type(solver).__name__}")

    policy = PrismHeadPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
        head_ckpt=head_ckpt,
        jepa_object_path=jepa_object,
        injection_mode=injection_mode,
        sigma_scale=sigma_scale,
        device="cuda",
        verbose=True,
    )

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent

    # ---- Sample eval episodes (same logic as eval.py) ----
    ep_indices = np.unique(
        dataset.get_col_data(
            "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        )
    )
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(f"{valid_mask.sum()} valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    elapsed = time.time() - start_time

    sr = metrics["success_rate"]
    print(f"\n=== PRISM-HEAD RESULT ({injection_mode}, sigma_scale={sigma_scale}) ===")
    print(f"success_rate: {sr}")
    print(f"wall time:    {elapsed:.1f}s")
    print(f"n_eval:       {cfg.eval.num_eval}")

    # One row per run in results/<task>.tsv (shared with eval.py).
    solver_name = "mppi" if "MPPI" in solver_target else "cem"
    method = {"pog": "prism", "none": "vanilla", "bc_only": "mean_only"}.get(injection_mode, injection_mode)
    out = REPO / "results" / f"{task}.tsv"
    out.parent.mkdir(exist_ok=True)
    new = not out.exists()
    with out.open("a") as f:
        if new:
            f.write("task\tsolver\tmethod\tK\tsigma_scale\tseed\tnum_eval\tsuccess_rate\n")
        f.write(f"{task}\t{solver_name}\t{method}\t{cfg.solver.num_samples}\t{sigma_scale}\t"
                f"{cfg.seed}\t{cfg.eval.num_eval}\t{sr:.2f}\n")
    print(f"appended to {out}")


if __name__ == "__main__":
    run()
