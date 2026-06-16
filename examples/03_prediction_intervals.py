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
        conformal_coverage=0.90,
        backend="auto",
        device="auto",
        random_state=42,
    )

    model.fit(X[:-12], y[:-12])

    pred, lower, upper = model.predict_interval(X[-12:])

    print("Prediction intervals")
    print("--------------------")

    for i in range(len(pred)):
        print(
            f"sample={i:02d}",
            f"pred={pred[i]:.6f}",
            f"lower={lower[i]:.6f}",
            f"upper={upper[i]:.6f}",
        )

    print("Conformal radius:", model.trace_.conformal_abs_radius)


if __name__ == "__main__":
    main()
