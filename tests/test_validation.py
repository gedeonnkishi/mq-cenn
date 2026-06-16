import numpy as np
import pytest

from mq_cenn.utils.validation import (
    as_float64,
    as_1d_float64,
    as_2d_float64,
    safe_std,
    _as_float64,
    _as_1d_float64,
    _as_2d_float64,
    _safe_std,
)


def test_as_float64_converts_to_numpy_float64():
    arr = as_float64([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64


def test_as_float64_rejects_nan():
    with pytest.raises(ValueError):
        as_float64([1.0, np.nan, 3.0])


def test_as_2d_float64_accepts_matrix():
    X = as_2d_float64([[1, 2], [3, 4]])
    assert X.shape == (2, 2)
    assert X.dtype == np.float64


def test_as_2d_float64_rejects_1d_array():
    with pytest.raises(ValueError):
        as_2d_float64([1, 2, 3])


def test_as_1d_float64_flattens_column_vector():
    y = as_1d_float64([[1], [2], [3]])
    assert y.shape == (3,)


def test_safe_std_adds_epsilon():
    X = np.ones((5, 2))
    s = safe_std(X, axis=0)
    assert np.all(s > 0)


def test_backward_compatible_aliases_exist():
    assert _as_float64 is as_float64
    assert _as_1d_float64 is as_1d_float64
    assert _as_2d_float64 is as_2d_float64
    assert _safe_std is safe_std
