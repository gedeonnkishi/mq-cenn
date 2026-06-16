from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal


BackendName = Literal["auto", "numpy", "cpp", "cuda"]
DeviceName = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class BackendInfo:
    backend: str
    device: str
    cuda_available: bool
    cpp_available: bool
    cuda_backend_available: bool


def is_torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def is_torch_cuda_available() -> bool:
    if not is_torch_available():
        return False

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_cpp_backend_available() -> bool:
    """
    Check whether the compiled C++ backend is available.

    This will return False until the native C++ extension is implemented
    and successfully installed.
    """
    return importlib.util.find_spec("mq_cenn._cpp_backend") is not None


def is_cuda_backend_available() -> bool:
    """
    Check whether the compiled CUDA backend is available.

    This will return False until the native CUDA extension is implemented
    and successfully installed.
    """
    return importlib.util.find_spec("mq_cenn._cuda_backend") is not None


def resolve_device(device: DeviceName = "auto") -> str:
    """
    Resolve execution device.

    - auto  -> cuda if available, otherwise cpu
    - cpu   -> force CPU
    - cuda  -> require CUDA
    """
    if device == "cpu":
        return "cpu"

    if device == "cuda":
        if not is_torch_cuda_available():
            raise RuntimeError(
                "device='cuda' was requested, but CUDA is not available. "
                "Use device='cpu' or device='auto'."
            )
        return "cuda"

    if device == "auto":
        return "cuda" if is_torch_cuda_available() else "cpu"

    raise ValueError(f"Unknown device: {device!r}")


def resolve_backend(
    backend: BackendName = "auto",
    device: DeviceName = "auto",
) -> BackendInfo:
    """
    Resolve the best available backend.

    Priority for backend='auto':
    1. CUDA backend, only if CUDA is available and compiled CUDA backend exists.
    2. C++ backend, only if compiled C++ backend exists.
    3. NumPy backend, always available.
    """
    cuda_available = is_torch_cuda_available()
    cpp_available = is_cpp_backend_available()
    cuda_backend_available = is_cuda_backend_available()

    resolved_device = resolve_device(device)

    if backend == "numpy":
        return BackendInfo(
            backend="numpy",
            device="cpu" if resolved_device == "cpu" else resolved_device,
            cuda_available=cuda_available,
            cpp_available=cpp_available,
            cuda_backend_available=cuda_backend_available,
        )

    if backend == "cpp":
        if not cpp_available:
            raise RuntimeError(
                "backend='cpp' was requested, but the C++ backend is not installed."
            )

        return BackendInfo(
            backend="cpp",
            device="cpu",
            cuda_available=cuda_available,
            cpp_available=cpp_available,
            cuda_backend_available=cuda_backend_available,
        )

    if backend == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "backend='cuda' was requested, but CUDA is not available."
            )

        if not cuda_backend_available:
            raise RuntimeError(
                "backend='cuda' was requested, but the CUDA backend is not installed."
            )

        return BackendInfo(
            backend="cuda",
            device="cuda",
            cuda_available=cuda_available,
            cpp_available=cpp_available,
            cuda_backend_available=cuda_backend_available,
        )

    if backend == "auto":
        if cuda_available and cuda_backend_available:
            return BackendInfo(
                backend="cuda",
                device="cuda",
                cuda_available=cuda_available,
                cpp_available=cpp_available,
                cuda_backend_available=cuda_backend_available,
            )

        if cpp_available:
            return BackendInfo(
                backend="cpp",
                device="cpu",
                cuda_available=cuda_available,
                cpp_available=cpp_available,
                cuda_backend_available=cuda_backend_available,
            )

        return BackendInfo(
            backend="numpy",
            device="cpu",
            cuda_available=cuda_available,
            cpp_available=cpp_available,
            cuda_backend_available=cuda_backend_available,
        )

    raise ValueError(f"Unknown backend: {backend!r}")


__all__ = [
    "BackendInfo",
    "resolve_backend",
    "resolve_device",
    "is_torch_available",
    "is_torch_cuda_available",
    "is_cpp_backend_available",
    "is_cuda_backend_available",
]
