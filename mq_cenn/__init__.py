"""
MQ-CeNN public API.

This package currently re-exports the legacy single-file implementation.
The code will be migrated module by module.
"""

from .legacy import (
    KernelSpec,
    DEFAULT_KERNEL_SPECS,
    MQCeNNRegressor,
    MQCeNNTrace,
    make_ablation_suite,
    set_global_seed,
)

try:
    from .legacy import __version__ as __version__
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "KernelSpec",
    "DEFAULT_KERNEL_SPECS",
    "MQCeNNRegressor",
    "MQCeNNTrace",
    "make_ablation_suite",
    "set_global_seed",
]
