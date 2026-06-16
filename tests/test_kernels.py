import numpy as np
import pytest

from mq_cenn.core.kernels import (
    DEFAULT_KERNEL_SPECS,
    KernelSpec,
    SpectralFeatureProjector,
)


def test_default_kernel_specs_not_empty():
    assert len(DEFAULT_KERNEL_SPECS) >= 1


def test_kernel_spec_is_immutable():
    spec = KernelSpec("gaussian", gamma=1.0)
    with pytest.raises(Exception):
        spec.gamma = 2.0


@pytest.mark.parametrize(
    "spec",
    [
        KernelSpec("gaussian"),
        KernelSpec("matern32"),
        KernelSpec("laplacian"),
        KernelSpec("periodic", period=12.0),
        KernelSpec("polynomial", degree=2),
    ],
)
def test_projector_fit_transform_shape(spec):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 4))

    projector = SpectralFeatureProjector(
        spec=spec,
        n_features=16,
        random_state=42,
    )

    Z = projector.fit_transform(X)

    assert Z.shape == (12, 16)
    assert np.isfinite(Z).all()


def test_projector_rejects_unsupported_kernel():
    with pytest.raises(ValueError):
        SpectralFeatureProjector(KernelSpec("unknown"), n_features=16)


def test_projector_rejects_small_feature_count():
    with pytest.raises(ValueError):
        SpectralFeatureProjector(KernelSpec("gaussian"), n_features=2)


def test_transform_before_fit_raises_error():
    X = np.random.randn(5, 3)
    projector = SpectralFeatureProjector(KernelSpec("gaussian"), n_features=16)

    with pytest.raises(RuntimeError):
        projector.transform(X)


def test_transform_rejects_wrong_feature_dimension():
    X = np.random.randn(10, 4)
    X_bad = np.random.randn(10, 5)

    projector = SpectralFeatureProjector(KernelSpec("gaussian"), n_features=16)
    projector.fit(X.shape[1])

    with pytest.raises(ValueError):
        projector.transform(X_bad)
