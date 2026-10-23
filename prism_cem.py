"""prism_cem.py — CEMSolver subclass supporting prior injection (paper Table VI).

  • injection_mode='warm_start'
      Use the prior μ as the initial sampling mean; σ_p is discarded. CEM then
      re-fits (mean, std) from the elite samples every iteration.

  • injection_mode='pog'
      PoG-fuse (μ_p, σ_p) into the initial sampling distribution with the same
      closed form as PrismMPPISolver. Unlike MPPI, CEM then adapts σ from the
      elites, so the prior's precision only shapes the first iteration.

  • injection_mode='none'
      No injection — identical to upstream CEMSolver.

Note on naming: upstream CEMSolver.init_action_distrib returns a tensor
called `var` but it's actually std (used as `mean + noise * std` ~ N(μ, σ²)).
Same mislabel as upstream MPPISolver. We treat it as std throughout.
"""
import torch
from stable_worldmodel.solver import CEMSolver

from prism_mppi import pog_fusion


class PrismCEMSolver(CEMSolver):
    """CEM with optional prior injection (warm-start, PoG, or none).

    injection_mode dispatch:
      'warm_start' : replace init mean with μ_p; ignore σ_p; keep default σ.
                     (Original behavior — backward compatible.)
      'pog'        : PoG-fuse (mean, std) with (μ_p, σ_p · prior_sigma_scale)
                     at init time. After init, CEM still updates σ from elites
                     — so this differs from PrismMPPI's frozen σ.
      'none'       : no injection (sanity / passthrough).

    set_prior(μ, σ) stores BOTH; the dispatch above decides which to use.
    """

    VALID_MODES = ("pog", "warm_start", "none")

    def __init__(
        self,
        *args,
        injection_mode: str = "warm_start",
        sigma_floor: float = 0.05,
        prior_sigma_scale: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert injection_mode in self.VALID_MODES, (
            f"injection_mode must be one of {self.VALID_MODES}, "
            f"got {injection_mode!r}"
        )
        self.injection_mode = injection_mode
        self.sigma_floor = sigma_floor
        self.prior_sigma_scale = prior_sigma_scale
        self._prior_mean = None
        self._prior_sigma = None
        self.verbose = verbose
        self._first_init_call = True

    def set_prior(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> None:
        """Store the prior μ and σ for the next solve() call.

        σ is kept in absolute units (already-scaled if user pre-multiplied).
        It is only consumed by injection_mode='pog'; 'warm_start' ignores it.
        """
        self._prior_mean = mu
        if sigma is None:
            # broadcastable default; only matters for 'pog'
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

        if self._prior_mean is None or self.injection_mode == "none":
            return mean, std

        mu_p = self._prior_mean.to(mean.device, dtype=mean.dtype)
        if mu_p.shape != mean.shape:
            mu_p = mu_p.expand_as(mean)

        if self.injection_mode == "warm_start":
            if self.verbose and self._first_init_call:
                print(f"[PrismCEM/warm_start] pre-inject  mean[0,0,:5] = "
                      f"{mean[0, 0, :5].cpu().tolist()}")
                print(f"[PrismCEM/warm_start] post-inject mean[0,0,:5] = "
                      f"{mu_p[0, 0, :5].cpu().tolist()}")
                print(f"[PrismCEM/warm_start] std unchanged at var_scale="
                      f"{self.var_scale}; σ_p IGNORED")
                self._first_init_call = False
            return mu_p, std

        elif self.injection_mode == "pog":
            assert self._prior_sigma is not None, (
                "injection_mode='pog' requires σ_p — call set_prior(μ, σ)"
            )
            sg_p = self._prior_sigma.to(std.device, dtype=std.dtype)
            if sg_p.shape != std.shape:
                sg_p = sg_p.expand_as(std)

            mean_post, std_post = pog_fusion(
                mean, std, mu_p, sg_p, self.sigma_floor
            )

            if self.verbose and self._first_init_call:
                print(f"[PrismCEM/pog] pre-PoG  mean[0,0,:5] = "
                      f"{mean[0, 0, :5].cpu().tolist()}")
                print(f"[PrismCEM/pog] pre-PoG  std [0,0,:5] = "
                      f"{std[0, 0, :5].cpu().tolist()}")
                print(f"[PrismCEM/pog] post-PoG mean[0,0,:5] = "
                      f"{mean_post[0, 0, :5].cpu().tolist()}")
                print(f"[PrismCEM/pog] post-PoG std [0,0,:5] = "
                      f"{std_post[0, 0, :5].cpu().tolist()}")
                print(f"[PrismCEM/pog] (CEM will adapt σ from elites in "
                      f"subsequent iterations — that's the key contrast "
                      f"with PrismMPPI/pog where σ stays frozen.)")
                self._first_init_call = False

            return mean_post, std_post

        raise RuntimeError(f"Unknown injection_mode: {self.injection_mode!r}")
