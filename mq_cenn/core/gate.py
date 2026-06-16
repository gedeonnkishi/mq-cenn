from __future__ import annotations

import torch
import torch.nn as nn


class SignedInterferenceGate(nn.Module):
    """
    Temporal gate with signed or softmax expert weighting.

    signed=True:
        The gate returns signed L1-normalized weights. This allows
        cancellation-like behavior between experts.

    signed=False:
        The gate returns softmax weights. This is useful as an ablation
        baseline because softmax cannot assign negative weights.
    """

    def __init__(
        self,
        context_channels: int,
        n_experts: int,
        bridge_dim: int = 32,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.05,
        signed: bool = True,
    ) -> None:
        super().__init__()

        self.context_channels = int(context_channels)
        self.n_experts = int(n_experts)
        self.bridge_dim = int(bridge_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.signed = bool(signed)

        if self.context_channels < 1:
            raise ValueError("context_channels must be >= 1.")

        if self.n_experts < 1:
            raise ValueError("n_experts must be >= 1.")

        if self.bridge_dim < 1:
            raise ValueError("bridge_dim must be >= 1.")

        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1.")

        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1.")

        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        padding = self.kernel_size // 2

        self.temporal = nn.Sequential(
            nn.Conv1d(
                self.context_channels,
                self.hidden_dim,
                self.kernel_size,
                padding=padding,
            ),
            nn.Tanh(),
            nn.Conv1d(
                self.hidden_dim,
                self.hidden_dim,
                self.kernel_size,
                padding=padding,
            ),
            nn.Tanh(),
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.bridge_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_experts),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        bridge: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_seq:
            Tensor of shape (batch_size, sequence_length, context_channels).

        bridge:
            Tensor of shape (batch_size, bridge_dim).

        Returns
        -------
        torch.Tensor
            Expert weights of shape (batch_size, n_experts).
        """
        if x_seq.ndim != 3:
            raise ValueError(
                "x_seq must be a 3D tensor with shape "
                "(batch_size, sequence_length, context_channels)."
            )

        if bridge.ndim != 2:
            raise ValueError(
                "bridge must be a 2D tensor with shape "
                "(batch_size, bridge_dim)."
            )

        if x_seq.shape[0] != bridge.shape[0]:
            raise ValueError("x_seq and bridge batch sizes must match.")

        if x_seq.shape[2] != self.context_channels:
            raise ValueError(
                f"Expected context_channels={self.context_channels}, "
                f"got {x_seq.shape[2]}."
            )

        if bridge.shape[1] != self.bridge_dim:
            raise ValueError(
                f"Expected bridge_dim={self.bridge_dim}, got {bridge.shape[1]}."
            )

        h = self.temporal(x_seq.permute(0, 2, 1)).mean(dim=-1)
        logits = self.head(torch.cat([h, bridge], dim=-1))

        if not self.signed:
            return torch.softmax(logits, dim=-1)

        raw = torch.tanh(logits)
        denom = raw.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return raw / denom


__all__ = [
    "SignedInterferenceGate",
]
