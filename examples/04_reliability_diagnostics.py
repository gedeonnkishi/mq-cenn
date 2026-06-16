import numpy as np

from mq_cenn import MQCeNNRegressor


def make_dataset(n=140, lookback=24, seed=42):
    rng = np.random.default_rng(seed)

    t = np.arange(n)
    series = np.sin(t / 6.0) + 0.05 * rng.normal(size=n)

    X = []
    y = []

    for i in range(lookback, n):
        X.append(series[i - lookback:i])
        y.append(series[i])

    return np.asarray(X), np.asarray(y)


def main():
    lookback = 24
    X, y = make_dataset(lookback=lookback)

    model = MQCeNNRegressor(
        n_features_per_expert=32,
        n_experts_per_kernel=1,
        bridge_dim=8,
        cenn_hidden=8,
        cenn_epochs=1,
        batch_size=16,
        patience=1,
        stationarize=True,
        last_value_index=lookback - 1,
        reliability_threshold=0.30,
        backend="auto",
        device="auto",
        random_state=42,
    )

    model.fit(X[:-12], y[:-12])

    diagnostics = model.predict_with_diagnostics(X[-12:])

    print("Diagnostics keys:", diagnostics.keys())
    print("Prediction shape:", diagnostics["prediction"].shape)
    print("Reliability:", diagnostics["reliability"])
    print("Fallback mask:", diagnostics["fallback_mask"])
    print("Teacher mean:", diagnostics["teacher_mean"])
    print("Pool predictions shape:", diagnostics["pool_predictions"].shape)

    print("\nTrace")
    print("-----")
    print("Backend:", model.trace_.backend)
    print("Device:", model.trace_.device)
    print("Mean calibration reliability:", model.trace_.mean_reliability_cal)
    print("Fallback rate calibration:", model.trace_.fallback_rate_cal)


if __name__ == "__main__":
    main()
