from __future__ import annotations

from typing import Dict, Mapping, Optional

from mq_cenn.core.kernels import DEFAULT_KERNEL_SPECS, KernelSpec
from mq_cenn.estimators import MQCeNNRegressor


def _copy_params(
    base_model: Optional[MQCeNNRegressor] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """
    Build a clean parameter dictionary for ablation variants.

    Parameters
    ----------
    base_model:
        Optional MQCeNNRegressor instance used as the template.
        If None, default MQCeNNRegressor parameters are used.

    overrides:
        Parameters to override for one ablation variant.

    Returns
    -------
    dict
        Parameter dictionary compatible with MQCeNNRegressor.
    """
    if base_model is None:
        params = MQCeNNRegressor().get_params(deep=False)
    else:
        params = base_model.get_params(deep=False)

    params = dict(params)

    if overrides:
        params.update(dict(overrides))

    return params


def make_ablation_suite(
    base_model: Optional[MQCeNNRegressor] = None,
    **overrides: object,
) -> Dict[str, MQCeNNRegressor]:
    """
    Create a reproducible ablation suite for MQ-CeNN.

    The suite is designed for scientific comparison. It does not select
    the best model automatically. Each returned estimator must be trained
    and evaluated with the same data split, metrics and random seed.

    Parameters
    ----------
    base_model:
        Optional MQCeNNRegressor used as a configuration template.

    **overrides:
        Common parameter overrides applied to all variants.

    Returns
    -------
    dict[str, MQCeNNRegressor]
        Dictionary of named ablation estimators.

    Variants
    --------
    MQCeNN_full:
        Full model with all default components.

    MQCeNN_softmax_gate:
        Replaces signed interference weights with softmax weights.

    MQCeNN_no_periodic_kernel:
        Removes the periodic kernel family.

    MQCeNN_gaussian_only:
        Uses only the Gaussian kernel family.

    MQCeNN_strict_reliability:
        Uses a higher reliability threshold, increasing fallback usage.

    MQCeNN_no_fallback:
        Disables fallback by setting the reliability threshold below zero.
    """
    common = _copy_params(base_model, overrides)

    kernel_specs = tuple(common.get("kernel_specs", DEFAULT_KERNEL_SPECS))

    non_periodic_specs = tuple(
        spec for spec in kernel_specs if spec.name != "periodic"
    )

    if not non_periodic_specs:
        non_periodic_specs = (
            KernelSpec("gaussian", gamma=1.0),
        )

    gaussian_only_specs = (
        KernelSpec("gaussian", gamma=1.0),
    )

    suite = {
        "MQCeNN_full": MQCeNNRegressor(
            **common,
        ),
        "MQCeNN_softmax_gate": MQCeNNRegressor(
            **_copy_params(
                base_model,
                {
                    **common,
                    "signed_interference": False,
                },
            )
        ),
        "MQCeNN_no_periodic_kernel": MQCeNNRegressor(
            **_copy_params(
                base_model,
                {
                    **common,
                    "kernel_specs": non_periodic_specs,
                },
            )
        ),
        "MQCeNN_gaussian_only": MQCeNNRegressor(
            **_copy_params(
                base_model,
                {
                    **common,
                    "kernel_specs": gaussian_only_specs,
                },
            )
        ),
        "MQCeNN_strict_reliability": MQCeNNRegressor(
            **_copy_params(
                base_model,
                {
                    **common,
                    "reliability_threshold": max(
                        0.50,
                        float(common.get("reliability_threshold", 0.30)),
                    ),
                },
            )
        ),
        "MQCeNN_no_fallback": MQCeNNRegressor(
            **_copy_params(
                base_model,
                {
                    **common,
                    "reliability_threshold": -1.0,
                },
            )
        ),
    }

    return suite


__all__ = [
    "make_ablation_suite",
]
