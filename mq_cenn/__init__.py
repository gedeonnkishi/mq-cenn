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

    # Estimators
    "FallbackStrategy": "mq_cenn.estimators",
    "MQCeNNRegressor": "mq_cenn.estimators",
    "MQCeNNTrace": "mq_cenn.estimators",

    # Ablation
    "make_ablation_suite": "mq_cenn.ablation",

    # Utilities
    "set_global_seed": "mq_cenn.utils",
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
    "make_ablation_suite",
    "set_global_seed",
]
