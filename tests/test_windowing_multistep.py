import numpy as np

from mq_cenn.preprocessing.windowing import make_multistep_windows


def test_make_multistep_windows_shapes_and_metadata_univariate():
    series = np.arange(20, dtype=float)

    X, Y, meta = make_multistep_windows(
        series,
        lookback=5,
        horizon=3,
        flatten=True,
        return_metadata=True,
    )

    assert X.shape == (13, 5)
    assert Y.shape == (13, 3)
    assert meta.last_value_index == 4
    np.testing.assert_array_equal(X[0], np.array([0, 1, 2, 3, 4], dtype=float))
    np.testing.assert_array_equal(Y[0], np.array([5, 6, 7], dtype=float))


def test_make_multistep_windows_metadata_multivariate_flattened():
    values = np.column_stack([
        np.arange(30, dtype=float),
        100 + np.arange(30, dtype=float),
    ])

    X, Y, meta = make_multistep_windows(
        values,
        target_index=0,
        lookback=4,
        horizon=2,
        flatten=True,
        return_metadata=True,
    )

    assert X.shape[1] == 8
    assert Y.shape[1] == 2
    assert meta.n_features == 2
    assert meta.last_value_index == 6
    assert X[0, meta.last_value_index] == 3.0
