"""PrismHeadPolicy — drives MPPI/CEM sampling with a learned head's Gaussian prior.

Sources (μ, σ) from a frozen PriorHead operating on JEPA-encoded
(current_obs, goal_obs) and hands them to a PRISM solver.

injection_mode dispatch (set at policy construction):
  'pog'        — solver.set_prior(μ, σ); PoG fusion at init time. PRISM.
  'warm_start' — solver.set_prior(μ); planner uses μ as init mean and its default σ.
                 The head's σ is computed but not used (warm-start baseline).
  'none'       — upstream planner as-is; the head is still loaded so all modes share
                 the same wiring.
  'bc_only'    — mean-only execution (paper Table II): bypass the planner. The head's μ
                 fills the action buffer, popped one env step at a time, and the head is
                 re-queried with a fresh z_t every flatten_receding_horizon steps.

The goal embedding is JEPA-encoded once per eval (LeWM evals run all episodes
in parallel envs; each env's goal is fixed for its episode), then cached.
Call `reset_goal_cache()` between distinct evals.
"""
from __future__ import annotations

import numpy as np
import torch
from stable_worldmodel.policy import WorldModelPolicy

from prior_head import PriorHead


# ImageNet normalization — must match eval.py's img_transform exactly.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class PrismHeadPolicy(WorldModelPolicy):
    VALID_MODES = ("none", "warm_start", "pog", "bc_only")

    def __init__(
        self,
        solver,
        config,
        process=None,
        transform=None,
        head_ckpt: str = None,
        jepa_object_path: str = None,
        injection_mode: str = "pog",
        sigma_scale: float = 1.0,
        device: str = "cuda",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(solver=solver, config=config, process=process,
                         transform=transform, **kwargs)
        assert injection_mode in self.VALID_MODES, \
            f"injection_mode must be one of {self.VALID_MODES}, got {injection_mode!r}"
        assert head_ckpt is not None, "Must pass head_ckpt"
        assert jepa_object_path is not None, "Must pass jepa_object_path"
        assert hasattr(solver, "set_prior"), \
            "Solver must implement set_prior — use PrismMPPISolver or PrismCEMSolver"

        self.injection_mode = injection_mode
        self.sigma_scale = sigma_scale
        self.device = device
        self.verbose = verbose

        # ---- Load frozen JEPA encoder (same encoder/projector used by world model) ----
        jepa = torch.load(jepa_object_path, map_location=device, weights_only=False)
        jepa.eval()
        for p in jepa.parameters():
            p.requires_grad_(False)
        self._jepa_encoder = jepa.encoder
        self._jepa_projector = jepa.projector

        # ---- Load frozen PriorHead ----
        ckpt = torch.load(head_ckpt, map_location=device, weights_only=False)
        head_cfg = ckpt["config"]
        self.head = PriorHead(
            z_dim=head_cfg["z_dim"], H=head_cfg["H"],
            A_block=head_cfg["A_block"], A_raw=head_cfg["A_raw"],
            hidden=head_cfg["hidden"], sigma_floor=head_cfg["sigma_floor"],
        ).to(device)
        self.head.load_state_dict(ckpt["state_dict"])
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self._H = head_cfg["H"]
        self._A_block = head_cfg["A_block"]
        self._A_raw = head_cfg["A_raw"]

        # ---- Configure solver mode ----
        # bc_only ⇒ solver runs in 'none' mode (we bypass it anyway, but if any
        # code accidentally calls solve(), it must NOT apply fusion).
        solver_mode = "none" if injection_mode == "bc_only" else injection_mode
        self.solver.injection_mode = solver_mode
        self.solver.prior_sigma_scale = sigma_scale  # consumed by 'pog'

        # ---- Episode-level cache ----
        self._goal_z = None  # (n_envs, z_dim)

        # ---- Device-resident ImageNet stats ----
        self._im_mean = _IMAGENET_MEAN.to(device)
        self._im_std = _IMAGENET_STD.to(device)

        if verbose:
            print(f"[PrismHead] injection_mode={injection_mode}  sigma_scale={sigma_scale}  "
                  f"(solver.injection_mode={solver_mode})  head_dim={head_cfg['z_dim']}")

    # -- Public hooks --

    def reset_goal_cache(self):
        """Clear the cached goal embedding. Call between distinct evals."""
        self._goal_z = None

    # -- Encoding helpers (operate on side-channel copies — never mutate info_dict) --

    def _normalize_pixels(self, raw: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Accept (n_envs, T, H, W, 3) OR (n_envs, H, W, 3) uint8/float, return
        (n_envs, T, 3, H, W) ImageNet-normalized float32 on self.device.

        Matches eval.py's img_transform (ToImage + ToDtype scale=True + Normalize(ImageNet)).
        No Resize — the task datasets already provide 224×224.
        """
        if torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)
        if arr.ndim == 4:           # (n_envs, H, W, 3) → add singleton T
            arr = arr[:, None]
        # arr now (n_envs, T, H, W, 3)
        if arr.dtype == np.uint8:
            t = torch.from_numpy(arr).to(self.device).permute(0, 1, 4, 2, 3).float().div_(255.0)
        else:
            t = torch.from_numpy(arr).to(self.device).permute(0, 1, 4, 2, 3).float()
            # heuristic: if max > 1.5, it was uint8-cast-to-float; else assume already 0..1
            if t.max() > 1.5:
                t = t / 255.0
        return (t - self._im_mean) / self._im_std

    def _jepa_encode(self, normalized: torch.Tensor) -> torch.Tensor:
        """(n_envs, T, 3, H, W) → (n_envs, z_dim) — last-timestep CLS + projector."""
        n_envs, T = normalized.shape[:2]
        x = normalized.reshape(n_envs * T, *normalized.shape[2:])
        with torch.no_grad():
            out = self._jepa_encoder(x, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]
            emb = self._jepa_projector(cls)  # (n_envs * T, z_dim)
        emb = emb.reshape(n_envs, T, -1)
        return emb[:, -1]  # last timestep = current observation

    def _ensure_goal_z(self, info_dict: dict):
        if self._goal_z is None:
            normalized = self._normalize_pixels(info_dict["goal"])
            self._goal_z = self._jepa_encode(normalized)
            if self.verbose:
                print(f"[PrismHead] goal_z cached: shape={tuple(self._goal_z.shape)}")

    def _compute_z_t(self, info_dict: dict) -> torch.Tensor:
        return self._jepa_encode(self._normalize_pixels(info_dict["pixels"]))

    def _run_head(self, z_t: torch.Tensor, z_g: torch.Tensor):
        """Returns (mu, sigma) shaped (n_envs, H, A_block * A_raw) for solver.set_prior."""
        with torch.no_grad():
            mu, sigma = self.head(z_t, z_g)  # (B, H, A_block, A_raw)
        n_envs = mu.size(0)
        return (
            mu.reshape(n_envs, self._H, self._A_block * self._A_raw),
            sigma.reshape(n_envs, self._H, self._A_block * self._A_raw),
        )

    # -- Main entry --

    def get_action(self, info_dict: dict, **kwargs):
        if self.injection_mode == "bc_only":
            return self._get_action_bc_only(info_dict, **kwargs)

        # All planner-using modes: encode + run head + dispatch via solver.
        z_t = self._compute_z_t(info_dict)
        self._ensure_goal_z(info_dict)
        mu, sigma = self._run_head(z_t, self._goal_z)

        if self.injection_mode == "none":
            self.solver.clear_prior()  # ensure no stale prior leaks across calls
        else:
            self.solver.set_prior(mu, sigma)

        return super().get_action(info_dict, **kwargs)

    def _get_action_bc_only(self, info_dict: dict, **kwargs):
        """Mean-only execution. Mirrors WorldModelPolicy.get_action's buffer logic.

        Replan (= re-query head with fresh z_t) every flatten_receding_horizon env
        steps; pop one env-step's action per call; inverse_transform at output.
        """
        if len(self._action_buffer) == 0:
            z_t = self._compute_z_t(info_dict)
            self._ensure_goal_z(info_dict)
            with torch.no_grad():
                mu, _sigma_unused = self.head(z_t, self._goal_z)
            n_envs = self.env.num_envs
            # mu: (n_envs, H, A_block, A_raw) → (n_envs, H * A_block, A_raw)
            plan = mu.reshape(n_envs, self.flatten_receding_horizon, self._A_raw).cpu()
            # Mirror upstream: deque receives one tensor per env-step, each shape (n_envs, A_raw)
            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action
