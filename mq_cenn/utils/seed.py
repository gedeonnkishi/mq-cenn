from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(
    seed: int,
    *,
    deterministic_torch: bool = False,
    env_hash_seed: Optional[bool] = None,
) -> None:
    """
    Set global random seeds for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed:
        Integer seed value.
    deterministic_torch:
        If True, asks PyTorch to use deterministic algorithms when possible.
        This can improve reproducibility but may reduce performance.
    env_hash_seed:
        If True, sets PYTHONHASHSEED in the current process environment.
        This is mostly useful when set before the Python interpreter starts.
        If None, it is left unchanged.

    Notes
    -----
    This function does not guarantee perfect bitwise reproducibility across
    different hardware, CUDA versions, BLAS backends or PyTorch versions.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if env_hash_seed is True:
        os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


__all__ = [
    "set_global_seed",
]
