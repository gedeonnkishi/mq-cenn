from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from types import ModuleType
from typing import Optional


CPP_EXTENSION_NAME = "mq_cenn._cpp_backend"


def is_available() -> bool:
    """
    Return True if the compiled C++ extension is installed.
    """
    return importlib.util.find_spec(CPP_EXTENSION_NAME) is not None


def load_cpp_extension() -> ModuleType:
    """
    Load the compiled C++ backend extension.

    This will work only after the native C++ extension is implemented
    and installed.
    """
    if not is_available():
        raise RuntimeError(
            "The C++ backend is not available. "
            "The extension 'mq_cenn._cpp_backend' has not been built yet."
        )

    return importlib.import_module(CPP_EXTENSION_NAME)


@dataclass(frozen=True)
class CppBackend:
    """
    Wrapper around the future compiled C++ backend.

    This class gives the Python package a stable interface before the native
    implementation exists.
    """

    name: str = "cpp"
    supports_cuda: bool = False
    supports_autograd: bool = False

    def module(self) -> ModuleType:
        return load_cpp_extension()


_CPP_BACKEND: Optional[CppBackend] = None


def get_cpp_backend() -> CppBackend:
    if not is_available():
        raise RuntimeError(
            "C++ backend requested, but it is not installed. "
            "Use backend='numpy' or backend='auto'."
        )

    global _CPP_BACKEND

    if _CPP_BACKEND is None:
        _CPP_BACKEND = CppBackend()

    return _CPP_BACKEND


__all__ = [
    "CPP_EXTENSION_NAME",
    "CppBackend",
    "is_available",
    "load_cpp_extension",
    "get_cpp_backend",
]
