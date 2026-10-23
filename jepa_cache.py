"""
Cache frozen LeWM-JEPA features for every frame of a task's h5 dataset, plus one goal
embedding per episode (its last frame). train_prior_head.py trains on this cache.

Output:
    $STABLEWM_HOME/jepa_cache/<task>/
        features.npy   # (N_frames, D) fp16 memmap
        goals.npy      # (N_episodes, D) fp16
        meta.json

Run:
    python jepa_cache.py --task cube
    python jepa_cache.py --task cube --batch 256
    python jepa_cache.py --task cube --dry-run     # first 5 episodes only
"""
import argparse
import json
import os
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  — registers Blosc codec
import numpy as np
import torch

# Dataset and world-model checkpoint per task, relative to STABLEWM_HOME.
TASKS = {
    "pusht":   {"h5": "pusht_expert_train.h5",         "ckpt": "pusht/lewm_object.ckpt"},
    "cube":    {"h5": "ogbench/cube_single_expert.h5", "ckpt": "cube/lewm_object.ckpt"},
    "tworoom": {"h5": "tworoom.h5",                    "ckpt": "tworooms/lewm_object.ckpt"},
    "reacher": {"h5": "dmc/reacher_random.h5",         "ckpt": "reacher/lewm_object.ckpt"},
}

# ImageNet normalization constants — matches eval.py's img_transform.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_uint8_batch(arr_uint8: np.ndarray, device) -> torch.Tensor:
    """(N, H, W, 3) uint8 → (N, 3, H, W) float32 on device, ImageNet-normalized."""
    t = torch.from_numpy(arr_uint8).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    return (t - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true",
                    help="Process only the first 5 episodes to verify the schema.")
    args = ap.parse_args()

    stable_home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    cfg = TASKS[args.task]
    h5_path = stable_home / cfg["h5"]
    ckpt_path = stable_home / cfg["ckpt"]
    out_dir = stable_home / "jepa_cache" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[jepa-cache] task={args.task}")
    print(f"[jepa-cache] h5  ={h5_path}")
    print(f"[jepa-cache] ckpt={ckpt_path}")
    print(f"[jepa-cache] out ={out_dir}")
    print(f"[jepa-cache] device={device}  batch={args.batch}  dry={args.dry_run}")

    print("\n[jepa-cache] loading JEPA module…")
    t0 = time.time()
    jepa = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    print(f"[jepa-cache] loaded in {time.time()-t0:.1f}s")

    encoder = jepa.encoder
    projector = jepa.projector
    with torch.no_grad():  # probe the embedding size with one dummy frame
        out = encoder(torch.zeros(1, 3, 224, 224, device=device), interpolate_pos_encoding=True)
        hidden_dim = projector(out.last_hidden_state[:, 0]).shape[-1]
    print(f"[jepa-cache] hidden_dim={hidden_dim}")

    print(f"\n[jepa-cache] opening {h5_path}…")
    with h5py.File(h5_path, "r") as f:
        n_frames = f["pixels"].shape[0]
        ep_offset = f["ep_offset"][:]
        ep_len = f["ep_len"][:]
        n_episodes = len(ep_len)
        n_episodes_use = 5 if args.dry_run else n_episodes

        # Frames to encode = all frames of episodes [0, n_episodes_use).
        n_target = int(ep_offset[n_episodes_use - 1] + ep_len[n_episodes_use - 1])
        print(f"[jepa-cache] n_episodes={n_episodes}  n_frames_total={n_frames}  "
              f"using {n_episodes_use} eps / {n_target} frames")

        # Goal = last frame of each episode.
        goal_indices = (ep_offset[:n_episodes_use] + ep_len[:n_episodes_use] - 1).astype(np.int64)
        assert goal_indices.max() < n_target

        feat_path = out_dir / ("features_dry.npy" if args.dry_run else "features.npy")
        features = np.lib.format.open_memmap(
            str(feat_path), mode="w+", dtype=np.float16, shape=(n_target, hidden_dim),
        )
        print(f"[jepa-cache] feature memmap: {feat_path} "
              f"({n_target} × {hidden_dim} fp16 = {n_target * hidden_dim * 2 / 1e9:.2f} GB)")

        t_start = time.time()
        n_done = 0
        with torch.no_grad():
            for lo in range(0, n_target, args.batch):
                hi = min(lo + args.batch, n_target)
                inputs = normalize_uint8_batch(f["pixels"][lo:hi], device)  # contiguous slice
                out = encoder(inputs, interpolate_pos_encoding=True)
                emb = projector(out.last_hidden_state[:, 0])  # (b, hidden_dim)
                features[lo:hi] = emb.to(torch.float16).cpu().numpy()
                n_done += hi - lo
                if (lo // args.batch) % 50 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done / max(elapsed, 1e-6)
                    print(f"[jepa-cache] {n_done:>7d}/{n_target} ({100*n_done/n_target:5.1f}%) "
                          f"rate={rate:.0f} fr/s  ETA={(n_target - n_done) / max(rate, 1e-6) / 60:.1f} min")

    elapsed = time.time() - t_start
    print(f"\n[jepa-cache] features done. {n_done} frames in {elapsed/60:.1f} min "
          f"({n_done/elapsed:.0f} fr/s)")

    goals = features[goal_indices].copy()
    goals_path = out_dir / ("goals_dry.npy" if args.dry_run else "goals.npy")
    np.save(goals_path, goals)
    print(f"[jepa-cache] goals.npy: shape={goals.shape}  → {goals_path}")

    # ---- Sanity checks on a random sample of features ----
    rng = np.random.RandomState(0)
    sample = np.asarray(features[rng.choice(n_target, size=min(2048, n_target), replace=False)],
                        dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    n_dead = int((sample.std(axis=0) < 1e-4).sum())
    ok = np.isfinite(sample).all() and n_dead <= hidden_dim // 2 and 1e-3 < norms.mean() < 1e6
    print(f"\n[jepa-cache] feature L2 norm: min={norms.min():.3f} max={norms.max():.3f} "
          f"mean={norms.mean():.3f}; dead dims (std<1e-4): {n_dead}/{hidden_dim}; "
          f"{'OK' if ok else 'SUSPICIOUS — check the checkpoint and dataset'}")

    meta = {
        "task": args.task,
        "h5_path": str(h5_path),
        "ckpt_path": str(ckpt_path),
        "hidden_dim": int(hidden_dim),
        "n_frames": int(n_done),
        "n_episodes": int(n_episodes_use),
        "goal_mode": "endframe",
        "dtype": "float16",
        "preprocessing": "uint8/255 → ImageNet mean/std",
        "elapsed_sec": float(elapsed),
        "dry_run": bool(args.dry_run),
    }
    meta_path = out_dir / ("meta_dry.json" if args.dry_run else "meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[jepa-cache] wrote {meta_path}")
    del features  # flush memmap


if __name__ == "__main__":
    main()
