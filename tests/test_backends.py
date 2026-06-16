import numpy as np
import pytest

from mq_cenn.backends import (
    get_numpy_backend,
    resolve_backend,
    resolve_device,
)
from mq_cenn.backends.cpp_backend import is_available as cpp_is_available
from mq_cenn.backends.cuda_backend import is_available as cuda_is_available


def test_resolve_backend_auto_returns_backend_info():
    info = resolve_backend("auto", "auto")

    assert info.backend in {"numpy", "cpp", "cuda"}
    assert info.device in {"cpu", "cuda"}
    assert isinstance(info.cuda_available, bool)
    assert isinstance(info.cpp_available, bool)
    assert isinstance(info.cuda_backend_available, bool)


def test_resolve_backend_numpy_is_always_available():
    info = resolve_backend("numpy", "cpu")

    assert info.backend == "numpy"
    assert info.device == "cpu"


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_unknown_raises_error():
    with pytest.raises(ValueError):
        resolve_device("invalid")


def test_numpy_backend_ridge_solve_shape():
    rng = np.random.default_rng(42)
    Z = rng.normal(size=(20, 5))
    y = rng.normal(size=20)

    backend = get_numpy_backend()
    beta = backend.ridge_solve(Z, y, alpha=1.0)

    assert beta.shape == (5,)
    assert np.isfinite(beta).all()


def test_native_backends_report_boolean_availability():
    assert isinstance(cpp_is_available(), bool)
    assert isinstance(cuda_is_available(), bool)
