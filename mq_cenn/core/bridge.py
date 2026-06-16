from __future__ import annotations

import torch
import torch.nn as nn


class CrossExpertBridge(nn.Module):
    """
    Pairwise interaction bridge over expert predictions.

    The bridge receives normalized expert predictions and enriches them with
    pairwise cross-expert interactions before the gate.

    This is an operational, classical proxy for cross-expert dependence.
    It is not a claim of physical quantum entanglement.
    """

    def __init__(
        self,
        n_experts: int,
        bridge_dim: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.n_experts = int(n_experts)
        self.bridge_dim = int(bridge_dim)
        self.dropout = float(dropout)

        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1.")

        if self.bridge_dim < 1:
            raise ValueError("bridge_dim must be >= 1.")

        cross_dim = self.n_experts * self.n_experts

        self.net = nn.Sequential(
            nn.Linear(cross_dim + self.n_experts, self.bridge_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.bridge_dim, self.bridge_dim),
            nn.Tanh(),
            nn.LayerNorm(self.bridge_dim),
        )

    def forward(self, pool_preds_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pool_preds_norm:
            Tensor of shape (batch_size, n_experts).

        Returns
        -------
        torch.Tensor
            Bridge representation of shape (batch_size, bridge_dim).
        """
        if pool_preds_norm.ndim != 2:
            raise ValueError(
                "pool_preds_norm must be a 2D tensor with shape "
                "(batch_size, n_experts)."
            )

        if pool_preds_norm.shape[1] != self.n_experts:
            raise ValueError(
                f"Expected {self.n_experts} experts, got "
                f"{pool_preds_norm.shape[1]}."
            )

        outer = torch.bmm(
            pool_preds_norm.unsqueeze(2),
            pool_preds_norm.unsqueeze(1),
        ).reshape(pool_preds_norm.shape[0], -1)

        z = torch.cat([pool_preds_norm, outer], dim=-1)

        return self.net(z)


__all__ = [
    "CrossExpertBridge",
]
