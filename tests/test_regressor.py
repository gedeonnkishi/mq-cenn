import numpy as np
import pytest

from mq_cenn import MQCeNNRegressor, MQCeNNTrace


def make_series_dataset(n=80, lookback=12, seed=42):
    rng = np.random.default_rng(seed)
    series = np.sin(np.arange(n) / 5.0) + 0.03 * rng.normal(size=n)

    X = []
    y = []

    for i in range(lookback, n):
        X.append(series[i - lookback:i])
        y.append(series[i])

    return np.asarray(X), np.asarray(y)


def make_fast_model(**overrides):
    params = dict(
        n_features_per_expert=16,
        n_experts_per_kernel=1,
        bridge_dim=8,
        cenn_hidden=8,
        cenn_epochs=1,
        batch_size=16,
        patience=1,
        stationarize=True,
        last_value_index=11,
        backend="auto",
        device="auto",
        random_state=42,
    )

    params.update(overrides)

    return MQCeNNRegressor(**params)


def test_regressor_fit_predict_shape():
    X, y = make_series_dataset()

    model = make_fast_model()
    model.fit(X[:-8], y[:-8])

    pred = model.predict(X[-8:])

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert isinstance(model.trace_, MQCeNNTrace)


def test_regressor_trace_contains_backend_and_device():
    X, y = make_series_dataset()

    model = make_fast_model()
    model.fit(X[:-8], y[:-8])

    assert model.trace_.backend in {"numpy", "cpp", "cuda"}
    assert model.trace_.device in {"cpu", "cuda"}


def test_predict_interval_shapes():
    X, y = make_series_dataset()

    model = make_fast_model()
    model.fit(X[:-8], y[:-8])

    pred, lower, upper = model.predict_interval(X[-8:])

    assert pred.shape == (8,)
    assert lower.shape == (8,)
    assert upper.shape == (8,)


def test_predict_with_diagnostics_keys():
    X, y = make_series_dataset()

    model = make_fast_model()
    model.fit(X[:-8], y[:-8])

    d = model.predict_with_diagnostics(X[-8:])

    expected = {
        "prediction",
        "core_prediction",
        "reliability",
        "fallback_mask",
        "interval_lower",
        "interval_upper",
        "teacher_mean",
        "pool_predictions",
    }

    assert expected.issubset(d.keys())
    assert d["prediction"].shape == (8,)
    assert d["reliability"].shape == (8,)


def test_predict_before_fit_raises_error():
    X, _ = make_series_dataset()

    model = make_fast_model()

    with pytest.raises(RuntimeError):
        model.predict(X[:5])
