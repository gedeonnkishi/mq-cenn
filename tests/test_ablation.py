import numpy as np

from mq_cenn import MQCeNNRegressor, make_ablation_suite


def make_series_dataset(n=70, lookback=10, seed=42):
    rng = np.random.default_rng(seed)
    series = np.sin(np.arange(n) / 5.0) + 0.03 * rng.normal(size=n)

    X = []
    y = []

    for i in range(lookback, n):
        X.append(series[i - lookback:i])
        y.append(series[i])

    return np.asarray(X), np.asarray(y)


def test_make_ablation_suite_keys():
    suite = make_ablation_suite(
        n_features_per_expert=16,
        n_experts_per_kernel=1,
        cenn_epochs=1,
        backend="auto",
        device="auto",
    )

    expected = {
        "MQCeNN_full",
        "MQCeNN_softmax_gate",
        "MQCeNN_no_periodic_kernel",
        "MQCeNN_gaussian_only",
        "MQCeNN_strict_reliability",
        "MQCeNN_no_fallback",
    }

    assert set(suite.keys()) == expected


def test_make_ablation_suite_values_are_regressors():
    suite = make_ablation_suite(
        n_features_per_expert=16,
        n_experts_per_kernel=1,
        cenn_epochs=1,
        backend="auto",
        device="auto",
    )

    assert all(isinstance(model, MQCeNNRegressor) for model in suite.values())


def test_one_ablation_model_can_fit_predict():
    X, y = make_series_dataset()

    suite = make_ablation_suite(
        n_features_per_expert=16,
        n_experts_per_kernel=1,
        bridge_dim=8,
        cenn_hidden=8,
        cenn_epochs=1,
        batch_size=16,
        patience=1,
        stationarize=True,
        last_value_index=9,
        backend="auto",
        device="auto",
        random_state=42,
    )

    model = suite["MQCeNN_full"]
    model.fit(X[:-8], y[:-8])

    pred = model.predict(X[-8:])

    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
