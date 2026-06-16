from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from types import ModuleType
from typing import Optional


CUDA_EXTENSION_NAME = "mq_cenn._cuda_backend"


def is_torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_available() -> bool:
    """
    Return True if both PyTorch CUDA and the compiled CUDA extension exist.
    """
    return (
        is_torch_cuda_available()
        and importlib.util.find_spec(CUDA_EXTENSION_NAME) is not None
    )


def load_cuda_extension() -> ModuleType:
    """
    Load the compiled CUDA backend extension.

    This will work only after the native CUDA extension is implemented,
    compiled and installed on a CUDA-compatible machine.
    """
    if not is_torch_cuda_available():
        raise RuntimeError(
            "CUDA backend requested, but PyTorch cannot access CUDA on this machine."
        )

    if importlib.util.find_spec(CUDA_EXTENSION_NAME) is None:
        raise RuntimeError(
            "CUDA backend requested, but the extension "
            "'mq_cenn._cuda_backend' has not been built yet."
        )

    return importlib.import_module(CUDA_EXTENSION_NAME)


@dataclass(frozen=True)
class CudaBackend:
    """
    Wrapper around the future compiled CUDA backend.
    """

    name: str = "cuda"
    supports_cuda: bool = True
    supports_autograd: bool = False

    def module(self) -> ModuleType:
        return load_cuda_extension()


_CUDA_BACKEND: Optional[CudaBackend] = None


def get_cuda_backend() -> CudaBackend:
    if not is_available():
        raise RuntimeError(
            "CUDA backend requested, but it is not available. "
            "Use backend='numpy', backend='cpp' or backend='auto'."
        )

    global _CUDA_BACKEND

    if _CUDA_BACKEND is None:
        _CUDA_BACKEND = CudaBackend()

    return _CUDA_BACKEND


__all__ = [
    "CUDA_EXTENSION_NAME",
    "CudaBackend",
    "is_torch_cuda_available",
    "is_available",
    "load_cuda_extension",
    "get_cuda_backend",
]
