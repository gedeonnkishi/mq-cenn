import numpy as np

from mq_cenn import MQCeNNRegressor


def make_dataset(n=120, lookback=24, seed=42):
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
        backend="auto",
        device="auto",
        random_state=42,
    )

    model.fit(X[:-10], y[:-10])
    pred = model.predict(X[-10:])

    print("Predictions shape:", pred.shape)
    print("Backend:", model.trace_.backend)
    print("Device:", model.trace_.device)
    print("Predictions:", pred)


if __name__ == "__main__":
    main()
