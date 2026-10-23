"""
Convert HuggingFace LeWM checkpoint to swm-compatible _object.ckpt.

Usage:
    python convert_hf_checkpoint.py <task>      # task in {pusht, cube, reacher, tworooms}

Reads:    $STABLEWM_HOME/hf_<task>/{config.json, weights.pt}
Writes:   $STABLEWM_HOME/<task>/lewm_object.ckpt
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_pretraining as spt
import stable_worldmodel as swm


def strip_target(d: dict) -> dict:
    """Remove Hydra `_target_` key from a config dict."""
    return {k: v for k, v in d.items() if k != "_target_"}


def convert(task: str) -> Path:
    """Convert HF weights to _object.ckpt for one task. Returns output path."""
    cache = Path(swm.data.utils.get_cache_dir())
    src = cache / f"hf_{task}"
    out = cache / task / "lewm_object.ckpt"

    print(f"\n=== Converting {task} ===")
    print(f"  src: {src}")
    print(f"  out: {out}")

    if not src.exists():
        raise FileNotFoundError(f"HF download dir not found: {src}")

    cfg = json.loads((src / "config.json").read_text())
    enc_cfg = strip_target(cfg["encoder"])
    encoder = spt.backbone.utils.vit_hf(**enc_cfg)

    def mlp(k):
        c = strip_target(cfg[k])
        c.pop("norm_fn", None)
        return MLP(**c, norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**strip_target(cfg["predictor"])),
        action_encoder=Embedder(**strip_target(cfg["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Saved ({out.stat().st_size / 1e6:.1f} MB, {n_params/1e6:.1f}M params)")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for task in sys.argv[1:]:
        try:
            convert(task)
        except Exception as e:
            print(f"  ✗ FAILED for {task}: {type(e).__name__}: {e}")
