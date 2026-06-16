import numpy as np
import pytest

from mq_cenn.core.experts import KernelRidgeExpert, MultiKernelExpertPool
from mq_cenn.core.kernels import KernelSpec


def make_regression_data(n=60, d=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    coef = rng.normal(size=d)
    y = X @ coef + 0.05 * rng.normal(size=n)
    return X, y


def test_kernel_ridge_expert_fit_predict_shape():
    X, y = make_regression_data()

    expert = KernelRidgeExpert(
        spec=KernelSpec("gaussian"),
        n_features=16,
        alpha=1.0,
        random_state=42,
    )

    expert.fit(X, y)
    pred = expert.predict(X[:7])

    assert pred.shape == (7,)
    assert np.isfinite(pred).all()


def test_kernel_ridge_expert_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        KernelRidgeExpert(
            spec=KernelSpec("gaussian"),
            n_features=16,
            alpha=0.0,
            random_state=42,
        )


def test_kernel_ridge_expert_predict_before_fit_raises_error():
    X, _ = make_regression_data()

    expert = KernelRidgeExpert(
        spec=KernelSpec("gaussian"),
        n_features=16,
        alpha=1.0,
        random_state=42,
    )

    with pytest.raises(RuntimeError):
        expert.predict(X)


def test_multi_kernel_expert_pool_fit_predict_pool_shape():
    X, y = make_regression_data(n=50)

    pool = MultiKernelExpertPool(
        kernel_specs=(KernelSpec("gaussian"), KernelSpec("laplacian")),
        n_experts_per_kernel=1,
        n_features_per_expert=16,
        alpha_grid=(0.1, 1.0),
        n_splits=2,
        random_state=42,
    )

    pool.fit(X, y)
    P = pool.predict_pool(X[:6])

    assert P.shape == (6, 2)
    assert np.isfinite(P).all()
    assert pool.n_experts_ == 2


def test_multi_kernel_expert_pool_predict_mean_shape():
    X, y = make_regression_data(n=50)

    pool = MultiKernelExpertPool(
        kernel_specs=(KernelSpec("gaussian"),),
        n_experts_per_kernel=2,
        n_features_per_expert=16,
        alpha_grid=(1.0,),
        random_state=42,
    )

    pool.fit(X, y)
    pred = pool.predict_mean(X[:5])

    assert pred.shape == (5,)
