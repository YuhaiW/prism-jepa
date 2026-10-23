"""PriorHead — MLP that maps (z_t, z_g) → Gaussian over an action sequence.

Paper Sec. III-B: input is concat(z_t, z_g) ∈ R^{2D}, output is (μ, σ) over
H × A_block × A_raw normalized actions. σ is per-input via softplus + a floor.
The head's output sits in StandardScaler-normalized action space; the eval-side
policy is responsible for inverse-transform back to env action units.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PriorHead(nn.Module):
    def __init__(
        self,
        z_dim: int,
        H: int,
        A_block: int,
        A_raw: int,
        hidden: int = 512,
        sigma_floor: float = 0.05,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.H = H
        self.A_block = A_block
        self.A_raw = A_raw
        self.action_seq_dim = H * A_block * A_raw
        self.sigma_floor = sigma_floor

        self.mlp = nn.Sequential(
            nn.Linear(2 * z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.action_seq_dim),
        )

    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor):
        """z_t, z_g: (B, D). Returns (mu, sigma) each of shape (B, H, A_block, A_raw)."""
        x = torch.cat([z_t, z_g], dim=-1)
        out = self.mlp(x)
        mu_flat, log_sigma_flat = out.chunk(2, dim=-1)
        sigma_flat = F.softplus(log_sigma_flat) + self.sigma_floor

        B = mu_flat.size(0)
        shape = (B, self.H, self.A_block, self.A_raw)
        return mu_flat.view(shape), sigma_flat.view(shape)


def beta_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """β-NLL (Seitzer et al. 2022).

    Standard Gaussian NLL per element (dropping additive constant):
        nll_i = 0.5 (y_i − μ_i)² / σ_i² + log σ_i
    β-NLL multiplies by stop_grad(σ_i^(2β)) before averaging:
        L = mean_i [ stop_grad(σ_i^{2β}) · nll_i ]
    β=0.5 is the recommended robust default — keeps σ-gradient alive but
    prevents the σ-blow-up pathology of vanilla NLL when μ is hard to fit.
    """
    var = sigma.pow(2)
    log_sigma = sigma.log()
    sq_err = (target - mu).pow(2)
    nll = 0.5 * sq_err / var + log_sigma
    weight = sigma.detach().pow(2 * beta)
    return (weight * nll).mean()
