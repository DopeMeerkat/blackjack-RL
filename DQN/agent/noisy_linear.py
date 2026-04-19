"""Factorized-Gaussian NoisyLinear layer.

Implements the NoisyNets recipe from Fortunato et al. (2017) using the
factorized Gaussian variant, as specified in blackjack_rl_design.md §15.1.

For a layer with input dim p and output dim q:

  Parameters (all learned by gradient descent):
    mu_w  : (q, p)   weight means
    sigma_w: (q, p)  weight noise scales
    mu_b  : (q,)     bias means
    sigma_b: (q,)    bias noise scales

  Noise (sampled per forward pass via reset_noise()):
    eps_in  ~ N(0, 1)^p,  transformed by f(x) = sign(x) * sqrt(|x|)
    eps_out ~ N(0, 1)^q,  transformed by f
    eps_W   = f(eps_out) ⊗ f(eps_in)    (outer product, shape q×p)
    eps_b   = f(eps_out)

  Forward (stochastic):
    W = mu_w + sigma_w ⊙ eps_W
    b = mu_b + sigma_b ⊙ eps_b
    y = W x + b

  Forward (deterministic / eval):
    y = mu_w x + mu_b

Initialization:
  mu    ~ Uniform[-1/√p, 1/√p]
  sigma  = sigma_0 / √p  (constant, sigma_0 = 0.5 by default)

Usage:
  layer = NoisyLinear(256, 256)
  layer.reset_noise()           # call before each forward pass during training
  out = layer(x)                # stochastic
  layer.set_deterministic(True)
  out = layer(x)                # deterministic (mean weights only)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized-Gaussian noisy linear layer (Fortunato et al., 2017)."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0
        self.deterministic = False   # toggled by set_deterministic()

        # Learnable parameters
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_b = nn.Parameter(torch.empty(out_features))
        self.sigma_b = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not parameters — moved with .to(device))
        self.register_buffer("eps_w", torch.zeros(out_features, in_features))
        self.register_buffer("eps_b", torch.zeros(out_features))

        self._init_parameters()
        self.reset_noise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.mu_w, -bound, bound)
        nn.init.uniform_(self.mu_b, -bound, bound)
        sigma_init = self.sigma0 / math.sqrt(self.in_features)
        nn.init.constant_(self.sigma_w, sigma_init)
        nn.init.constant_(self.sigma_b, sigma_init)

    # ------------------------------------------------------------------
    # Noise management
    # ------------------------------------------------------------------

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Squashing function f(x) = sign(x) * sqrt(|x|)."""
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Resample factorized noise.  Call before each training forward pass."""
        device = self.mu_w.device
        eps_in  = self._f(torch.randn(self.in_features,  device=device))
        eps_out = self._f(torch.randn(self.out_features, device=device))
        # Outer product: eps_W[i,j] = eps_out[i] * eps_in[j]
        self.eps_w.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.eps_b.copy_(eps_out)

    def set_deterministic(self, val: bool) -> None:
        """Switch between stochastic (val=False) and deterministic (val=True) mode."""
        self.deterministic = val

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deterministic:
            return F.linear(x, self.mu_w, self.mu_b)
        w = self.mu_w + self.sigma_w * self.eps_w
        b = self.mu_b + self.sigma_b * self.eps_b
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"sigma0={self.sigma0}"
        )
