from __future__ import annotations

from importlib import import_module


__version__ = "0.1.0"


_SYMBOL_TO_MODULE = {
    # Core kernels
    "KernelName": "mq_cenn.core.kernels",
    "KernelSpec": "mq_cenn.core.kernels",
    "DEFAULT_KERNEL_SPECS": "mq_cenn.core.kernels",
    "SpectralFeatureProjector": "mq_cenn.core.kernels",

    # Core experts
    "KernelRidgeExpert": "mq_cenn.core.experts",
    "MultiKernelExpertPool": "mq_cenn.core.experts",

    # Reliability
    "NoveltyDetector": "mq_cenn.core.reliability",
    "ReliabilityCalibrator": "mq_cenn.core.reliability",

    # Neural components
    "CrossExpertBridge": "mq_cenn.core.bridge",
    "SignedInterferenceGate": "mq_cenn.core.gate",

    # Estimators — single-step
    "FallbackStrategy": "mq_cenn.estimators.regressor",
    "MQCeNNRegressor": "mq_cenn.estimators.regressor",
    "MQCeNNTrace": "mq_cenn.estimators.regressor",

    # Estimators — multi-step
    "MQCeNNMultiStepRegressor": "mq_cenn.estimators.multistep",
    "MQCeNNMultiStepTrace": "mq_cenn.estimators.multistep",

    # Estimators — anomaly detection
    "MQCeNNAnomalyDetector": "mq_cenn.estimators.anomaly",
    "MQCeNNAnomalyTrace": "mq_cenn.estimators.anomaly",

    # Ablation
    "make_ablation_suite": "mq_cenn.ablation.suite",

    # Utilities
    "set_global_seed": "mq_cenn.utils.seed",

    # Preprocessing
    "make_supervised_windows": "mq_cenn.preprocessing.windowing",
    "make_multistep_windows": "mq_cenn.preprocessing.windowing",
    "chronological_split": "mq_cenn.preprocessing.windowing",
    "train_only_standardize": "mq_cenn.preprocessing.windowing",
    "flatten_windows": "mq_cenn.preprocessing.windowing",
    "make_calendar_features": "mq_cenn.preprocessing.calendar_features",
    "add_calendar_features": "mq_cenn.preprocessing.calendar_features",
    "add_seasonal_lag_features": "mq_cenn.preprocessing.seasonal_lags",
    "make_seasonal_lag_matrix": "mq_cenn.preprocessing.seasonal_lags",
    "seasonal_naive_forecast": "mq_cenn.preprocessing.seasonal_lags",
}


def __getattr__(name: str):
    """
    Lazy public API loader.

    This keeps lightweight imports fast while exposing the main framework API
    from the package root.
    """
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module 'mq_cenn' has no attribute {name!r}")

    module = import_module(_SYMBOL_TO_MODULE[name])
    value = getattr(module, name)

    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "KernelName",
    "KernelSpec",
    "DEFAULT_KERNEL_SPECS",
    "SpectralFeatureProjector",
    "KernelRidgeExpert",
    "MultiKernelExpertPool",
    "NoveltyDetector",
    "ReliabilityCalibrator",
    "CrossExpertBridge",
    "SignedInterferenceGate",
    "FallbackStrategy",
    "MQCeNNRegressor",
    "MQCeNNTrace",
    "MQCeNNMultiStepRegressor",
    "MQCeNNMultiStepTrace",
    "MQCeNNAnomalyDetector",
    "MQCeNNAnomalyTrace",
    "make_ablation_suite",
    "set_global_seed",
    "make_supervised_windows",
    "make_multistep_windows",
    "chronological_split",
    "train_only_standardize",
    "flatten_windows",
    "make_calendar_features",
    "add_calendar_features",
    "add_seasonal_lag_features",
    "make_seasonal_lag_matrix",
    "seasonal_naive_forecast",
]
