"""
MQ-CeNN public API.

The package exposes the main estimator API while keeping submodules
such as mq_cenn.utils importable without loading heavy ML dependencies.
"""

__version__ = "0.1.0"

_PUBLIC_LEGACY_SYMBOLS = {
    "KernelSpec",
    "DEFAULT_KERNEL_SPECS",
    "MQCeNNRegressor",
    "MQCeNNTrace",
    "make_ablation_suite",
    "set_global_seed",
}


def __getattr__(name: str):
    """
    Lazy-load legacy symbols only when they are explicitly requested.

    This prevents imports such as `from mq_cenn.utils import ...`
    from requiring scikit-learn or other heavy dependencies.
    """
    if name in _PUBLIC_LEGACY_SYMBOLS:
        from . import legacy

        value = getattr(legacy, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'mq_cenn' has no attribute '{name}'")


__all__ = [
    "__version__",
    "KernelSpec",
    "DEFAULT_KERNEL_SPECS",
    "MQCeNNRegressor",
    "MQCeNNTrace",
    "make_ablation_suite",
    "set_global_seed",
]
