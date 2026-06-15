from setuptools import setup
import os

# Sécurité si le README est manquant lors de certains tests locaux rapides
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="mq_cenn",
    version="1.0.0",
    py_modules=["mq_cenn"], # Cherche précisément mq_cenn.py à la racine
    install_requires=[
        "numpy>=1.22.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
    ],
    author="Gédéon Nkishi",
    description="Multi-Quantum Cellular Neural Network (MQ-CeNN) for Time Series Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gedeonnkishi/mq-cenn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
