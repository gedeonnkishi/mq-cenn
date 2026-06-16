import torch
import pytest

from mq_cenn.core.gate import SignedInterferenceGate


def test_signed_gate_forward_shape():
    gate = SignedInterferenceGate(
        context_channels=1,
        n_experts=3,
        bridge_dim=8,
        hidden_dim=8,
        dropout=0.0,
        signed=True,
    )

    x_seq = torch.randn(5, 10, 1)
    bridge = torch.randn(5, 8)

    weights = gate(x_seq, bridge)

    assert weights.shape == (5, 3)


def test_signed_gate_l1_normalization():
    gate = SignedInterferenceGate(
        context_channels=1,
        n_experts=3,
        bridge_dim=8,
        hidden_dim=8,
        dropout=0.0,
        signed=True,
    )

    x_seq = torch.randn(5, 10, 1)
    bridge = torch.randn(5, 8)

    weights = gate(x_seq, bridge)
    l1 = weights.abs().sum(dim=1)

    assert torch.allclose(l1, torch.ones_like(l1), atol=1e-5)


def test_softmax_gate_sums_to_one():
    gate = SignedInterferenceGate(
        context_channels=1,
        n_experts=3,
        bridge_dim=8,
        hidden_dim=8,
        dropout=0.0,
        signed=False,
    )

    x_seq = torch.randn(5, 10, 1)
    bridge = torch.randn(5, 8)

    weights = gate(x_seq, bridge)

    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-5)


def test_gate_rejects_wrong_context_channels():
    gate = SignedInterferenceGate(
        context_channels=1,
        n_experts=3,
        bridge_dim=8,
        hidden_dim=8,
    )

    x_seq = torch.randn(5, 10, 2)
    bridge = torch.randn(5, 8)

    with pytest.raises(ValueError):
        gate(x_seq, bridge)
