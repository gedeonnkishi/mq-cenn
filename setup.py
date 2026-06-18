from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent.resolve()


def env_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


BUILD_CPP = env_enabled("MQCENN_BUILD_CPP")
BUILD_CUDA = env_enabled("MQCENN_BUILD_CUDA")


def get_long_description() -> str:
    readme = ROOT / "README.md"

    if readme.exists():
        return readme.read_text(encoding="utf-8")

    return "MQ-CeNN framework for quantum-inspired time-series forecasting."


def get_cxx_compile_args() -> list[str]:
    if os.name == "nt":
        return ["/O2"]
    return ["-O3"]


def build_extensions():
    ext_modules = []
    cmdclass = {}

    include_dirs = [
        str(ROOT / "native" / "include"),
    ]

    if BUILD_CPP:
        try:
            from pybind11.setup_helpers import Pybind11Extension, build_ext
        except Exception as exc:
            raise RuntimeError(
                "C++ build was requested with MQCENN_BUILD_CPP=1, "
                "but pybind11 could not be imported. "
                "Install pybind11 first with: pip install pybind11"
            ) from exc

        ext_modules.append(
            Pybind11Extension(
                "mq_cenn._cpp_backend",
                sources=[
                    str(ROOT / "native" / "cpp" / "bindings.cpp"),
                    str(ROOT / "native" / "cpp" / "kernels.cpp"),
                    str(ROOT / "native" / "cpp" / "experts.cpp"),
                    str(ROOT / "native" / "cpp" / "utils.cpp"),
                ],
                include_dirs=include_dirs,
                cxx_std=17,
                extra_compile_args=get_cxx_compile_args(),
            )
        )

        cmdclass["build_ext"] = build_ext

    if BUILD_CUDA:
        try:
            from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        except Exception as exc:
            raise RuntimeError(
                "CUDA build was requested with MQCENN_BUILD_CUDA=1, "
                "but torch.utils.cpp_extension could not be imported. "
                "Install a CUDA-compatible PyTorch build first."
            ) from exc

        ext_modules.append(
            CUDAExtension(
                "mq_cenn._cuda_backend",
                sources=[
                    str(ROOT / "native" / "cuda" / "bindings_cuda.cpp"),
                    str(ROOT / "native" / "cuda" / "kernels_cuda.cu"),
                    str(ROOT / "native" / "cuda" / "experts_cuda.cu"),
                ],
                include_dirs=include_dirs,
                extra_compile_args={
                    "cxx": get_cxx_compile_args(),
                    "nvcc": ["-O3"],
                },
            )
        )

        cmdclass["build_ext"] = BuildExtension

    return ext_modules, cmdclass


ext_modules, cmdclass = build_extensions()


setup(
    name="mq-cenn",
    version="1.1.0",
    description="MQ-CeNN framework for quantum-inspired time-series forecasting",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Gedeon Nkishi",
    url="https://github.com/gedeonnkishi/mq-cenn",
    project_urls={
        "Source": "https://github.com/gedeonnkishi/mq-cenn",
        "Issues": "https://github.com/gedeonnkishi/mq-cenn/issues",
    },
    packages=find_packages(include=["mq_cenn", "mq_cenn.*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "build>=1.0",
            "wheel>=0.40",
            "pybind11>=2.11",
        ],
        "docs": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "notebook>=7.0",
            "ipykernel>=6.0",
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mqcenn=mq_cenn.cli.main:main",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
