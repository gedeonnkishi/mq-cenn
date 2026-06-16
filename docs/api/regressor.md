# Regressor API

The main public estimator is:

```python
from mq_cenn import MQCeNNRegressor
```

## `MQCeNNRegressor`

`MQCeNNRegressor` implements the main end-to-end regression model.

### Basic usage

```python
model = MQCeNNRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Main methods

#### `fit(X, y)`

Fits the MQ-CeNN model.

Parameters:

- `X`: two-dimensional array of shape `(n_samples, n_features)`;
- `y`: one-dimensional target array of shape `(n_samples,)`.

Returns:

- the fitted estimator.

#### `predict(X)`

Returns point predictions.

```python
y_pred = model.predict(X_test)
```

#### `predict_interval(X)`

Returns prediction intervals.

```python
pred, lower, upper = model.predict_interval(X_test)
```

The interval width is based on a conformal-style calibration radius.

#### `predict_with_diagnostics(X)`

Returns a dictionary containing predictions and internal diagnostic information.

Typical keys:

```text
prediction
core_prediction
reliability
fallback_mask
interval_lower
interval_upper
teacher_mean
pool_predictions
```

## Important parameters

### Model capacity

- `n_features_per_expert`: number of random features per expert;
- `n_experts_per_kernel`: number of experts per kernel family;
- `bridge_dim`: dimension of the cross-expert bridge representation;
- `cenn_hidden`: hidden dimension of the neural gate/bridge component.

### Training

- `cenn_epochs`: number of training epochs for the neural component;
- `batch_size`: mini-batch size;
- `patience`: early-stopping patience;
- `random_state`: seed for reproducibility.

### Forecasting

- `stationarize`: if enabled, the model works with residualized targets relative to the last observed value;
- `last_value_index`: index of the last observed value in each input window.

For time-series windows with lookback `W`, this is usually:

```python
last_value_index = W - 1
```

### Reliability

- `reliability_threshold`: threshold below which fallback may be used;
- `conformal_coverage`: nominal conformal coverage level.

### Backend

- `backend="auto"`: automatically selects the best available backend;
- `device="auto"`: automatically selects CPU or CUDA when supported.

On a CPU-only machine, `auto` usually resolves to:

```text
backend='numpy'
device='cpu'
```

## Trace object

After fitting, the estimator exposes:

```python
model.trace_
```

This contains useful metadata such as:

- backend;
- device;
- number of experts;
- validation loss;
- fallback rate;
- mean reliability;
- conformal radius.

## Scientific note

`MQCeNNRegressor` should always be evaluated against strong baselines. A good benchmark should include at least Persistence, Ridge, and tree-based regressors.
