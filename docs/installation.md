# Installation

This page explains how to install MQ-CeNN locally for development and experimentation.

## 1. Clone the repository

```bash
git clone https://github.com/gedeonnkishi/mq-cenn.git
cd mq-cenn
```

## 2. Create a virtual environment

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Upgrade packaging tools

MQ-CeNN uses `setuptools` and `wheel`. The version range below avoids conflicts with PyTorch environments that require `setuptools < 82`.

```bash
python -m pip install --upgrade "setuptools>=68,<82" wheel
```

## 4. Install in editable mode

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e .[dev]
```

## 5. Verify the installation

```bash
mqcenn doctor
```

Expected output should include:

```text
Python version
Platform
NumPy version
scikit-learn version
PyTorch availability
CUDA availability
Resolved backend
```

On a CPU-only machine, the expected resolved backend is usually:

```text
backend='numpy'
device='cpu'
```

## 6. Optional backend installation

MQ-CeNN is CPU-safe by default. Optional backends are handled explicitly.

```bash
mqcenn install-backend
mqcenn install-backend --cpu
mqcenn install-backend --cpp
mqcenn install-backend --cuda
```

The CUDA command does not install NVIDIA drivers, CUDA Toolkit, or a compiler. It only checks the environment and tries to build the optional CUDA extension if the system is ready.

## 7. Run tests

```bash
pytest
```

A healthy development installation should pass all unit tests.

## 8. Common installation issues

### The `mqcenn` command is not found

Reinstall the package in editable mode:

```bash
pip install -e .
```

Then open a new terminal or reactivate the virtual environment.

### CUDA is not detected

Check:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

If these commands fail, use the default CPU/NumPy backend.

### Do not commit the virtual environment

The `.venv/` directory must not be committed to GitHub. Add it to `.gitignore`.
