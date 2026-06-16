import pytest
import torch

from mq_cenn.core.bridge import CrossExpertBridge


def test_cross_expert_bridge_forward_shape():
    model = CrossExpertBridge(
        n_experts=3,
        bridge_dim=8,
        dropout=0.0,
    )

    x = torch.randn(5, 3)
    y = model(x)

    assert y.shape == (5, 8)


def test_cross_expert_bridge_rejects_wrong_rank():
    model = CrossExpertBridge(n_experts=3, bridge_dim=8)

    with pytest.raises(ValueError):
        model(torch.randn(5, 3, 1))


def test_cross_expert_bridge_rejects_wrong_expert_count():
    model = CrossExpertBridge(n_experts=3, bridge_dim=8)

    with pytest.raises(ValueError):
        model(torch.randn(5, 4))
