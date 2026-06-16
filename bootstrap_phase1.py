from pathlib import Path
import shutil

ROOT = Path(".").resolve()

# Paths
old_single_file = ROOT / "mq_cenn.py"
pkg_dir = ROOT / "mq_cenn"
legacy_file = pkg_dir / "legacy.py"
archive_dir = ROOT / "_legacy_archive"

# 1. Create folders
folders = [
    pkg_dir,
    pkg_dir / "utils",
    pkg_dir / "core",
    pkg_dir / "estimators",
    pkg_dir / "ablation",
    pkg_dir / "backends",
    ROOT / "native" / "cpp",
    ROOT / "native" / "cuda",
    ROOT / "native" / "include",
    ROOT / "tests",
    ROOT / "examples",
    ROOT / "docs",
    ROOT / "benchmarks",
    archive_dir,
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)

# 2. Preserve current single-file implementation as legacy
if old_single_file.exists():
    shutil.copy2(old_single_file, legacy_file)
    shutil.copy2(old_single_file, archive_dir / "mq_cenn_single_file_v1.py")
    old_single_file.unlink()
else:
    print("Warning: mq_cenn.py not found. Skipping legacy copy.")

# 3. Package __init__.py keeps public API working
(pkg_dir / "__init__.py").write_text(
    '''"""
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
''',
    encoding="utf-8",
)

# 4. utils package
(pkg_dir / "utils" / "__init__.py").write_text(
    '''from .validation import ArrayLike, _as_float64, _as_1d_float64, _as_2d_float64, _safe_std
from .seed import set_global_seed
from .conformal import _quantile_abs_residual

__all__ = [
    "ArrayLike",
    "_as_float64",
    "_as_1d_float64",
    "_as_2d_float64",
    "_safe_std",
    "set_global_seed",
    "_quantile_abs_residual",
]
''',
    encoding="utf-8",
)

(pkg_dir / "utils" / "validation.py").write_text(
    '''from __future__ import annotations

from typing import Sequence, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_float64(x: ArrayLike, *, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_2d_float64(x: ArrayLike, *, name: str = "X") -> np.ndarray:
    arr = _as_float64(x, name=name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    return arr


def _as_1d_float64(y: ArrayLike, *, name: str = "y") -> np.ndarray:
    arr = _as_float64(y, name=name).reshape(-1)
    return arr


def _safe_std(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    return np.std(x, axis=axis, keepdims=keepdims) + 1e-8
''',
    encoding="utf-8",
)

(pkg_dir / "utils" / "seed.py").write_text(
    '''from __future__ import annotations

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
''',
    encoding="utf-8",
)

(pkg_dir / "utils" / "conformal.py").write_text(
    '''from __future__ import annotations

import numpy as np


def _quantile_abs_residual(residuals: np.ndarray, coverage: float) -> float:
    """Split-conformal absolute residual quantile."""
    residuals = np.asarray(residuals, dtype=np.float64)

    if residuals.size == 0:
        return float("nan")

    coverage = float(np.clip(coverage, 0.50, 0.999))
    q = np.ceil((residuals.size + 1) * coverage) / residuals.size
    q = min(1.0, q)

    return float(np.quantile(np.abs(residuals), q))
''',
    encoding="utf-8",
)

# 5. Empty init files for future modules
for init_path in [
    pkg_dir / "core" / "__init__.py",
    pkg_dir / "estimators" / "__init__.py",
    pkg_dir / "ablation" / "__init__.py",
    pkg_dir / "backends" / "__init__.py",
]:
    init_path.write_text("", encoding="utf-8")

# 6. Backend placeholders
(pkg_dir / "backends" / "dispatcher.py").write_text(
    '''from __future__ import annotations


def get_backend(name: str = "auto") -> str:
    """
    Backend dispatcher placeholder.

    Current stable backend: numpy.
    Future backends: cpp, cuda.
    """
    if name in {"auto", "numpy"}:
        return "numpy"
    if name in {"cpp", "cuda"}:
        raise NotImplementedError(f"Backend '{name}' is planned but not implemented yet.")
    raise ValueError(f"Unknown backend: {name}")
''',
    encoding="utf-8",
)

(pkg_dir / "backends" / "numpy_backend.py").write_text(
    '''"""
Default NumPy backend.

This backend is always available and acts as the reference implementation.
"""
''',
    encoding="utf-8",
)

(pkg_dir / "backends" / "cpp_backend.py").write_text(
    '''"""
C++ backend wrapper placeholder.

The native implementation will be added after the Python framework is stable.
"""
''',
    encoding="utf-8",
)

(pkg_dir / "backends" / "cuda_backend.py").write_text(
    '''"""
CUDA backend wrapper placeholder.

The CUDA implementation will be added after the C++ backend is stable.
"""
''',
    encoding="utf-8",
)

# 7. Native placeholders
for placeholder in [
    ROOT / "native" / "cpp" / ".gitkeep",
    ROOT / "native" / "cuda" / ".gitkeep",
    ROOT / "native" / "include" / ".gitkeep",
    ROOT / "tests" / ".gitkeep",
    ROOT / "examples" / ".gitkeep",
    ROOT / "docs" / ".gitkeep",
    ROOT / "benchmarks" / ".gitkeep",
]:
    placeholder.write_text("", encoding="utf-8")

# 8. Backup and replace setup.py
setup_py = ROOT / "setup.py"

if setup_py.exists():
    shutil.copy2(setup_py, archive_dir / "setup_single_module_v1.py")

setup_py.write_text(
    '''from setuptools import setup, find_packages

setup(
    name="mq-cenn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.8.0",
    ],
    author="Gedeon Nkishi",
    description="MQ-CeNN framework for time-series forecasting",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gedeonnkishi/mq-cenn",
    python_requires=">=3.8",
)
''',
    encoding="utf-8",
)

print("Phase 1 completed successfully.")
print("Next: run `pip install -e .` and import tests.")
