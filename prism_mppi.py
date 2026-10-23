"""
PRISM-MPPI: MPPI solver with Product-of-Gaussians fusion of an external action prior.

Subclasses stable_worldmodel.solver.MPPISolver. Adds:
  - set_prior(mu, sigma): externally provide an action-prior Gaussian
  - PoG fusion is applied to (mean, std) right after init, before the MPPI iterations
"""

import torch
from stable_worldmodel.solver import MPPISolver


def pog_fusion(mean, std, mu_prior, sigma_prior, sigma_floor=0.05):
    """Product-of-Gaussians fusion of two diagonal Gaussians (paper Eqs. 4-5).

    Note: upstream MPPISolver.init_action_distrib returns a tensor named
    `var` but it is actually std — sampling is done as `noise * batch_var`
    (i.e. N(mean, std²)), not `noise * sqrt(batch_var)`. We treat that
    output as std throughout this file. PoG fusion below uses
    precision = 1/std².

    Args:
        mean:        (..., A) — MPPI initial mean
        std:         (..., A) — MPPI initial std (NOT variance, despite the
                                 stable-worldmodel naming convention)
        mu_prior:    (..., A) — external prior mean
        sigma_prior: (..., A) — external prior std
        sigma_floor: minimum output std

    Returns:
        fused_mean: (..., A)
        fused_std:  (..., A)
    """
    eps = 1e-8
    tau_base  = 1.0 / (std**2 + eps)
    tau_prior = 1.0 / (sigma_prior**2 + eps)
    tau_combined = tau_base + tau_prior

    fused_mean = (tau_base * mean + tau_prior * mu_prior) / tau_combined
    fused_std  = (1.0 / tau_combined).sqrt().clamp(min=sigma_floor)
    return fused_mean, fused_std


class PrismMPPISolver(MPPISolver):
    """MPPI with PoG fusion of an external action prior at init time.

    Identical to MPPISolver if no prior is set OR injection_mode='none'.
    The fused std is held fixed across the MPPI iterations (MPPI updates the mean only).

    injection_mode dispatch (only relevant when a prior IS set):
      'pog'        — Product-of-Gaussians fusion of N(μ_p, (s·σ_p)²)
                     with the planner default N(μ_default, σ_default²). PRISM-MPPI.
      'warm_start' — Replace the planner mean with μ_p; KEEP the planner default σ.
                     The head's σ_p is ignored (warm-start baseline).
      'none'       — Bypass any fusion (upstream MPPI).

    Args:
        sigma_floor:       minimum std after fusion
        prior_sigma_scale: prior scale s, multiplier on σ_p (only used by 'pog')
        injection_mode:    one of {'pog', 'warm_start', 'none'}
    """

    VALID_MODES = ('pog', 'warm_start', 'none')

    def __init__(
        self,
        *args,
        sigma_floor: float = 0.05,
        prior_sigma_scale: float = 1.0,
        injection_mode: str = 'pog',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert injection_mode in self.VALID_MODES, \
            f"injection_mode must be one of {self.VALID_MODES}, got {injection_mode!r}"
        self.sigma_floor = sigma_floor
        self.prior_sigma_scale = prior_sigma_scale
        self.injection_mode = injection_mode
        self._prior_mean = None   # (n_envs, horizon, action_dim) tensor
        self._prior_sigma = None  # same shape; absolute units (already scaled)

    def set_prior(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> None:
        """Set the action prior used at the next solve() call.

        Args:
            mu:    (n_envs, horizon, action_dim) — prior mean.
            sigma: same shape, OR scalar, OR None (in which case we use
                   prior_sigma_scale * 1.0 broadcast).
        """
        self._prior_mean = mu
        if sigma is None:
            sigma = torch.ones_like(mu) * self.prior_sigma_scale
        elif sigma.dim() == 0:
            sigma = torch.ones_like(mu) * sigma * self.prior_sigma_scale
        else:
            sigma = sigma * self.prior_sigma_scale
        self._prior_sigma = sigma

    def clear_prior(self) -> None:
        self._prior_mean = None
        self._prior_sigma = None

    def init_action_distrib(self, actions: torch.Tensor | None = None):
        """Initialize action distribution; dispatch on injection_mode."""
        mean, std = super().init_action_distrib(actions)

        # Fast bypass: no prior set, OR mode='none' → byte-identical to upstream.
        if self._prior_mean is None or self.injection_mode == 'none':
            return mean, std

        mu_p = self._prior_mean.to(mean.device, dtype=mean.dtype)
        if mu_p.shape != mean.shape:
            mu_p = mu_p.expand_as(mean)

        if self.injection_mode == 'warm_start':
            return mu_p, std

        # 'pog'
        assert self._prior_sigma is not None, "pog mode requires σ_p (set_prior with sigma)"
        sg_p = self._prior_sigma.to(std.device, dtype=std.dtype)
        if sg_p.shape != std.shape:
            sg_p = sg_p.expand_as(std)
        return pog_fusion(mean, std, mu_p, sg_p, self.sigma_floor)
