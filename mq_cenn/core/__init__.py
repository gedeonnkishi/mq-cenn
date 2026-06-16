from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "KernelName": "kernels",
    "KernelSpec": "kernels",
    "DEFAULT_KERNEL_SPECS": "kernels",
    "SpectralFeatureProjector": "kernels",
    "KernelRidgeExpert": "experts",
    "MultiKernelExpertPool": "experts",
    "NoveltyDetector": "reliability",
    "ReliabilityCalibrator": "reliability",
    "CrossExpertBridge": "bridge",
    "SignedInterferenceGate": "gate",
}


def __getattr__(name: str):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module 'mq_cenn.core' has no attribute {name!r}")

    module = import_module(f"mq_cenn.core.{_SYMBOL_TO_MODULE[name]}")
    value = getattr(module, name)
    globals()[name] = value

    return value


__all__ = list(_SYMBOL_TO_MODULE)
