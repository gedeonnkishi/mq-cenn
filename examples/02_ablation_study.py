import numpy as np

from mq_cenn import make_ablation_suite


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


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    lookback = 24
    X, y = make_dataset(lookback=lookback)

    X_train, y_train = X[:-10], y[:-10]
    X_test, y_test = X[-10:], y[-10:]

    suite = make_ablation_suite(
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

    print("Ablation results")
    print("----------------")

    for name, model in suite.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        print(
            name,
            "| MAE:",
            round(mae(y_test, pred), 6),
            "| RMSE:",
            round(rmse(y_test, pred), 6),
            "| backend:",
            model.trace_.backend,
            "| device:",
            model.trace_.device,
        )


if __name__ == "__main__":
    main()
