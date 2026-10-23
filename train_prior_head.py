"""Train the PRISM prior head on cached JEPA features (paper Sec. III-B).

Reads the feature cache written by jepa_cache.py and the actions of the task's h5
dataset, fits a StandardScaler matching eval.py (full action column, NaN-filtered),
and trains the head with β-NLL (β = 0.5) on an episode-level 90/10 train/val split.
The goal embedding is the last frame of each demonstration episode.

The head is small and (features, targets) fit on one GPU, so the data are moved to
the GPU once and batches are gathered by index.

Usage:
    python train_prior_head.py --task cube          # → checkpoints/prior_head_cube.pt

Diagnostics printed at the end (informational; the checkpoint is always saved):
  - best val MSE relative to the untrained head
  - whether train NLL decreased monotonically up to the best epoch
  - σ calibration: mean σ_p vs √(best val MSE)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  — registers Blosc codec
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim

from prior_head import PriorHead, beta_nll_loss


SEED = 42
H = 5          # planning steps
A_BLOCK = 5    # environment steps per planning step
WIN_LEN = H * A_BLOCK  # 25

EPOCHS = 50
BATCH = 256
LR = 3e-4
WD = 1e-4
WARMUP = 1000
BETA = 0.5

REPO = Path(__file__).resolve().parent
STABLE_HOME = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))

# Per-task dataset (relative to STABLEWM_HOME) and 90/10 episode-level train/val split.
TASKS = {
    # PushT: 15 epochs, the schedule of the released head (its best checkpoint is epoch 1 under 15 or 50 epochs).
    "pusht":   dict(h5="pusht_expert_train.h5",         train_ep_range=(0, 16817), val_ep_range=(16817, 18685), epochs=15),
    "cube":    dict(h5="ogbench/cube_single_expert.h5", train_ep_range=(0, 9000),  val_ep_range=(9000, 10000)),
    "tworoom": dict(h5="tworoom.h5",                    train_ep_range=(0, 9000),  val_ep_range=(9000, 10000)),
    "reacher": dict(h5="dmc/reacher_random.h5",         train_ep_range=(0, 9000),  val_ep_range=(9000, 10000)),
}


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_window_indices(ep_offset, ep_len, ep_lo, ep_hi, win_len=WIN_LEN):
    """Return (start_indices, episode_ids) — frame indices where a length-win_len action
    window can start without crossing into the NaN end-of-episode action."""
    starts_list = []
    eps_list = []
    for e in range(ep_lo, ep_hi):
        max_off = int(ep_len[e]) - win_len  # exclusive
        if max_off <= 0:
            continue
        offs = np.arange(max_off, dtype=np.int64)
        starts_list.append(int(ep_offset[e]) + offs)
        eps_list.append(np.full(max_off, e, dtype=np.int64))
    return np.concatenate(starts_list), np.concatenate(eps_list)


def cosine_warmup_factor(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--epochs", type=int, default=None, help=f"default: {EPOCHS} (PushT: 15)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--wd", type=float, default=WD)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--seed", type=int, default=SEED, help="RNG seed (default 42, the released heads)")
    ap.add_argument("--out", default=None, help="checkpoint path (default: checkpoints/prior_head_<task>.pt)")
    args = ap.parse_args()

    task_cfg = TASKS[args.task]
    if args.epochs is None:
        args.epochs = task_cfg.get("epochs", EPOCHS)
    h5_path = STABLE_HOME / task_cfg["h5"]
    cache_dir = STABLE_HOME / "jepa_cache" / args.task
    out_path = Path(args.out) if args.out else REPO / "checkpoints" / f"prior_head_{args.task}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train_ep_lo, train_ep_hi = task_cfg["train_ep_range"]
    val_ep_lo, val_ep_hi = task_cfg["val_ep_range"]

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] task={args.task}  device={device}  epochs={args.epochs}  "
          f"batch={args.batch}  lr={args.lr}  wd={args.wd}  beta={args.beta}  seed={args.seed}")
    print(f"[train] h5   ={h5_path}")
    print(f"[train] cache={cache_dir}")
    print(f"[train] out  ={out_path}")

    # ---- Load JEPA cache ----
    print("[train] loading JEPA cache…")
    t0 = time.time()
    features = np.load(cache_dir / "features.npy", mmap_mode="r")  # (N, D) fp16
    n_frames, D = features.shape
    goals = np.load(cache_dir / "goals.npy")  # (n_eps, D) fp16, last frame of each episode
    print(f"[train] features={features.shape} dtype={features.dtype}  "
          f"goals={goals.shape}  load={time.time()-t0:.1f}s")

    # ---- Load actions + episode metadata ----
    print("[train] loading actions + episode boundaries…")
    t0 = time.time()
    with h5py.File(h5_path, "r") as f:
        actions = f["action"][:].astype(np.float32)  # (N, A_raw)
        ep_offset = f["ep_offset"][:].astype(np.int64)
        ep_len = f["ep_len"][:].astype(np.int64)
    A_raw = actions.shape[1]
    print(f"[train] actions={actions.shape} A_raw={A_raw}  load={time.time()-t0:.1f}s")

    # ---- Fit StandardScaler on full action data (NaN-filtered) ----
    # eval.py fits its scaler on the full action column of the dataset; we match it
    # by fitting on the full h5 actions after dropping the end-of-episode NaN rows.
    print("[train] fitting StandardScaler on full action data (NaN-filtered)…")
    nan_mask = np.isnan(actions).any(axis=1)
    scaler = StandardScaler().fit(actions[~nan_mask])
    print(f"[train]   action mean : {scaler.mean_.round(4).tolist()}")
    print(f"[train]   action scale: {scaler.scale_.round(4).tolist()}")

    # ---- Build train/val window indices ----
    print("[train] building window indices…")
    t0 = time.time()
    train_starts, train_eps = build_window_indices(ep_offset, ep_len, train_ep_lo, train_ep_hi)
    val_starts, val_eps = build_window_indices(ep_offset, ep_len, val_ep_lo, val_ep_hi)
    print(f"[train]   train windows = {len(train_starts):>8d}")
    print(f"[train]   val   windows = {len(val_starts):>8d}  build={time.time()-t0:.1f}s")

    # ---- Pre-compute normalized action targets (M, H, A_block, A_raw) ----
    print("[train] pre-computing normalized action targets…")
    t0 = time.time()

    def compute_targets(starts):
        idx = starts[:, None] + np.arange(WIN_LEN, dtype=np.int64)[None, :]  # (M, WIN_LEN)
        seqs = actions[idx]  # (M, WIN_LEN, A_raw)
        flat_norm = scaler.transform(seqs.reshape(-1, A_raw)).astype(np.float32)
        return flat_norm.reshape(-1, H, A_BLOCK, A_raw)

    train_targets = compute_targets(train_starts)
    val_targets = compute_targets(val_starts)
    assert not np.isnan(train_targets).any(), "NaN slipped past filter (train)"
    assert not np.isnan(val_targets).any(), "NaN slipped past filter (val)"
    print(f"[train]   train_targets={train_targets.shape}")
    print(f"[train]   val_targets  ={val_targets.shape}  build={time.time()-t0:.1f}s")

    # ---- Move static data to GPU ----
    print("[train] moving tensors to GPU…")
    t0 = time.time()
    feats_gpu = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    goals_gpu = torch.from_numpy(np.asarray(goals, dtype=np.float32)).to(device)
    train_starts_gpu = torch.from_numpy(train_starts).to(device)
    train_eps_gpu = torch.from_numpy(train_eps).to(device)
    train_targets_gpu = torch.from_numpy(train_targets).to(device)
    val_starts_gpu = torch.from_numpy(val_starts).to(device)
    val_eps_gpu = torch.from_numpy(val_eps).to(device)
    val_targets_gpu = torch.from_numpy(val_targets).to(device)
    print(f"[train]   move={time.time()-t0:.1f}s")

    # ---- Build head ----
    head = PriorHead(z_dim=D, H=H, A_block=A_BLOCK, A_raw=A_raw).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[train] PriorHead params: {n_params/1e6:.2f}M  ({n_params:,})")

    opt = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    n_train = len(train_starts)
    steps_per_epoch = (n_train + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs
    print(f"[train] steps/epoch={steps_per_epoch}  total_steps={total_steps}")

    @torch.no_grad()
    def eval_val():
        head.eval()
        n_val = len(val_starts_gpu)
        nll_sum = mse_sum = 0.0
        n_done = 0
        for s in range(0, n_val, args.batch):
            e = min(s + args.batch, n_val)
            zt = feats_gpu[val_starts_gpu[s:e]]
            zg = goals_gpu[val_eps_gpu[s:e]]
            tgt = val_targets_gpu[s:e]
            mu, sg = head(zt, zg)
            count = e - s
            nll_sum += beta_nll_loss(mu, sg, tgt, beta=args.beta).item() * count
            mse_sum += ((mu - tgt) ** 2).mean().item() * count
            n_done += count
        head.train()
        return nll_sum / n_done, mse_sum / n_done

    init_val_nll, init_val_mse = eval_val()
    print(f"\n[train] EPOCH 0 (init): val NLL={init_val_nll:.4f}  val MSE={init_val_mse:.4f}")

    best_val_nll = float("inf")
    best_state = None
    best_epoch = 0
    history = []

    step = 0
    t_train_start = time.time()
    head.train()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_train, device=device)
        epoch_nll = 0.0
        n_seen = 0
        t_ep0 = time.time()
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            zt = feats_gpu[train_starts_gpu[idx]]
            zg = goals_gpu[train_eps_gpu[idx]]
            tgt = train_targets_gpu[idx]
            mu, sg = head(zt, zg)
            loss = beta_nll_loss(mu, sg, tgt, beta=args.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            lr_factor = cosine_warmup_factor(step, total_steps, WARMUP)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_factor
            opt.step()
            epoch_nll += loss.item() * idx.size(0)
            n_seen += idx.size(0)
            step += 1

        train_nll = epoch_nll / n_seen
        val_nll, val_mse = eval_val()
        ep_secs = time.time() - t_ep0
        history.append(dict(
            epoch=epoch, train_nll=train_nll, val_nll=val_nll,
            val_mse=val_mse, lr=args.lr * lr_factor, ep_secs=ep_secs,
        ))
        is_best = val_nll < best_val_nll
        if is_best:
            best_val_nll = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch

        if epoch <= 5 or epoch % 5 == 0 or epoch == args.epochs or is_best:
            print(f"[train] EPOCH {epoch:>2d}/{args.epochs}: "
                  f"train NLL={train_nll:>7.4f}  val NLL={val_nll:>7.4f}  "
                  f"val MSE={val_mse:.4f}  lr={args.lr * lr_factor:.2e}  "
                  f"ep={ep_secs:.1f}s{'  *best' if is_best else ''}")

    train_total = time.time() - t_train_start

    # ---- Diagnostics (informational) ----
    best_val_mse = min(h["val_mse"] for h in history)
    relative_drop = (init_val_mse - best_val_mse) / max(init_val_mse, 1e-9)
    print(f"\n=== Diagnostics ===")
    print(f"  val NLL: init {init_val_nll:.4f} → best {best_val_nll:.4f} @ epoch {best_epoch}")
    print(f"  val MSE: init {init_val_mse:.4f} → best {best_val_mse:.4f} "
          f"({100 * relative_drop:.1f}% lower)")
    if relative_drop < 0.15:
        print("  note: val MSE dropped by less than 15%. Expected on tasks whose long-offset "
              "actions are weakly predictable (TwoRoom, Reacher; paper Table I).")
    train_nll_up_to_best = [h["train_nll"] for h in history if h["epoch"] <= best_epoch]
    bumps = [i for i in range(1, len(train_nll_up_to_best))
             if train_nll_up_to_best[i] > train_nll_up_to_best[i - 1] + 5e-3]
    if bumps:
        print(f"  note: train NLL rose at epochs {[b + 1 for b in bumps]} before the best epoch.")

    head.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    @torch.no_grad()
    def sigma_on_val():
        head.eval()
        sigmas = []
        for s in range(0, len(val_starts_gpu), args.batch):
            e = min(s + args.batch, len(val_starts_gpu))
            _, sg = head(feats_gpu[val_starts_gpu[s:e]], goals_gpu[val_eps_gpu[s:e]])
            sigmas.append(sg.flatten().cpu())
        return torch.cat(sigmas)

    sigma_flat = sigma_on_val()
    sigma_floor = 0.05
    frac_near_floor = (sigma_flat <= sigma_floor * 1.05).float().mean().item()
    sigma_std = sigma_flat.std().item()
    print(f"  σ_p on val: mean {sigma_flat.mean().item():.4f}  std {sigma_std:.4f}  "
          f"min {sigma_flat.min().item():.4f}  max {sigma_flat.max().item():.4f}  "
          f"near floor {100 * frac_near_floor:.1f}%")
    print(f"  √(best val MSE) = {best_val_mse ** 0.5:.4f}  (mean σ_p of a calibrated head)")

    torch.save({
        "state_dict": best_state,
        "config": dict(z_dim=int(D), H=H, A_block=A_BLOCK, A_raw=A_raw,
                       hidden=512, sigma_floor=sigma_floor),
        "goal_mode": "endframe",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "best_val_nll": best_val_nll,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "init_val_nll": init_val_nll,
        "init_val_mse": init_val_mse,
        "history": history,
        "args": vars(args),
        "sigma_stats": dict(
            mean=float(sigma_flat.mean()), std=float(sigma_std),
            median=float(sigma_flat.median()),
            min=float(sigma_flat.min()), max=float(sigma_flat.max()),
            frac_near_floor=float(frac_near_floor),
        ),
    }, out_path)
    print(f"\n[train] saved best (val NLL={best_val_nll:.4f}, "
          f"val MSE={best_val_mse:.4f}, epoch {best_epoch}) → {out_path}")
    print(f"[train] total train time: {train_total/60:.1f} min")


if __name__ == "__main__":
    main()
