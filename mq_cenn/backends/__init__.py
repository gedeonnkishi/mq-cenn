from __future__ import annotations

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "BackendName": "dispatcher",
    "DeviceName": "dispatcher",
    "BackendInfo": "dispatcher",
    "resolve_backend": "dispatcher",
    "resolve_device": "dispatcher",
    "is_torch_available": "dispatcher",
    "is_torch_cuda_available": "dispatcher",
    "is_cpp_backend_available": "dispatcher",
    "is_cuda_backend_available": "dispatcher",
    "NumpyBackend": "numpy_backend",
    "get_numpy_backend": "numpy_backend",
    "CppBackend": "cpp_backend",
    "get_cpp_backend": "cpp_backend",
    "CudaBackend": "cuda_backend",
    "get_cuda_backend": "cuda_backend",
}


def __getattr__(name: str):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module 'mq_cenn.backends' has no attribute {name!r}")

    module = import_module(f"mq_cenn.backends.{_SYMBOL_TO_MODULE[name]}")
    value = getattr(module, name)

    globals()[name] = value
    return value


__all__ = list(_SYMBOL_TO_MODULE)
