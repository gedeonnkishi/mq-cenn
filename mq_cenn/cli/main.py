from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print("\nRunning:", " ".join(command))
    process = subprocess.run(command, env=env)
    return int(process.returncode)


def python_pip() -> list[str]:
    return [sys.executable, "-m", "pip"]


def install_cpu_backend() -> int:
    """
    Install the default CPU backend dependencies.

    This is the safest default backend. It works on CPU-only machines and
    does not require CUDA, NVIDIA drivers or a native compiler.
    """
    print("Installing MQ-CeNN CPU backend...")

    commands = [
        python_pip() + ["install", "--upgrade", "pip"],
        python_pip() + [
            "install",
            "numpy>=1.22.0",
            "scipy>=1.8.0",
            "scikit-learn>=1.0.0",
        ],
        python_pip() + [
            "install",
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
    ]

    for command in commands:
        code = run_command(command)
        if code != 0:
            return code

    print("\nCPU backend installed successfully.")
    return 0


def install_cpp_backend() -> int:
    """
    Install build dependencies and attempt to build the C++ backend.
    """
    print("Installing MQ-CeNN C++ backend...")

    commands = [
        python_pip() + ["install", "--upgrade", "pip"],
        python_pip() + ["install", "pybind11>=2.11", "wheel", "build"],
    ]

    for command in commands:
        code = run_command(command)
        if code != 0:
            return code

    env = os.environ.copy()
    env["MQCENN_BUILD_CPP"] = "1"

    code = run_command(python_pip() + ["install", "-e", "."], env=env)

    if code == 0:
        print("\nC++ backend built and installed successfully.")
    else:
        print(
            "\nC++ backend build failed. Make sure a C++17 compiler is installed."
        )

    return code


def install_cuda_backend() -> int:
    """
    Build the CUDA backend if the machine is CUDA-ready.

    This command does not silently install NVIDIA drivers or CUDA Toolkit.
    It checks whether the environment is ready, then attempts compilation.
    """
    print("Installing MQ-CeNN CUDA backend...")

    nvcc = shutil.which("nvcc")

    if nvcc is None:
        print(
            "\nCUDA Toolkit was not found: 'nvcc' is unavailable.\n"
            "Install NVIDIA CUDA Toolkit first, then rerun:\n"
            "mqcenn install-backend --cuda"
        )
        return 1

    try:
        import torch
    except ImportError:
        print(
            "\nPyTorch is not installed.\n"
            "Install a CUDA-compatible PyTorch build first, then rerun:\n"
            "mqcenn install-backend --cuda"
        )
        return 1

    if not torch.cuda.is_available():
        print(
            "\nPyTorch cannot access CUDA on this machine.\n"
            "Check GPU, NVIDIA driver, CUDA Toolkit and PyTorch CUDA build."
        )
        return 1

    code = run_command(
        python_pip() + ["install", "pybind11>=2.11", "wheel", "build"]
    )

    if code != 0:
        return code

    env = os.environ.copy()
    env["MQCENN_BUILD_CUDA"] = "1"

    code = run_command(python_pip() + ["install", "-e", "."], env=env)

    if code == 0:
        print("\nCUDA backend built and installed successfully.")
    else:
        print("\nCUDA backend build failed. Check CUDA Toolkit and compiler setup.")

    return code


def install_backend(args: argparse.Namespace) -> int:
    """
    Backend installation router.

    Default behavior:
        mqcenn install-backend

    is equivalent to:
        mqcenn install-backend --cpu
    """
    if args.cpp:
        return install_cpp_backend()

    if args.cuda:
        return install_cuda_backend()

    return install_cpu_backend()


def doctor(_: argparse.Namespace) -> int:
    """
    Print a local environment report.
    """
    print("MQ-CeNN environment report")
    print("--------------------------")
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print("Platform:", sys.platform)
    print("nvcc:", shutil.which("nvcc") or "not found")

    try:
        import numpy
        print("NumPy:", numpy.__version__)
    except Exception:
        print("NumPy: not installed")

    try:
        import scipy
        print("SciPy:", scipy.__version__)
    except Exception:
        print("SciPy: not installed")

    try:
        import sklearn
        print("scikit-learn:", sklearn.__version__)
    except Exception:
        print("scikit-learn: not installed")

    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA version:", torch.version.cuda)
    except Exception:
        print("PyTorch: not installed")

    try:
        from mq_cenn.backends import resolve_backend

        print("Resolved backend:", resolve_backend("auto", "auto"))
    except Exception as exc:
        print("Backend resolution failed:", exc)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mqcenn",
        description="MQ-CeNN command-line tools.",
    )

    subparsers = parser.add_subparsers(dest="command")

    install = subparsers.add_parser(
        "install-backend",
        help="Install or build a MQ-CeNN backend.",
    )

    group = install.add_mutually_exclusive_group()
    group.add_argument(
        "--cpu",
        action="store_true",
        help="Install the CPU backend. This is the default.",
    )
    group.add_argument(
        "--cpp",
        action="store_true",
        help="Build and install the C++ CPU backend.",
    )
    group.add_argument(
        "--cuda",
        action="store_true",
        help="Build and install the CUDA GPU backend.",
    )

    install.set_defaults(func=install_backend)

    doctor_cmd = subparsers.add_parser(
        "doctor",
        help="Inspect the local MQ-CeNN environment.",
    )
    doctor_cmd.set_defaults(func=doctor)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
