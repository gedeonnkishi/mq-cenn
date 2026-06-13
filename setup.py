from setuptools import setup, find_packages

setup(
    name="mq_cenn",
    version="1.0.0",
    py_modules=["mq_cenn"],
    install_requires=[
        "numpy>=1.22.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
    ],
    author="Gédéon Nkishi",
    description="Multi-Quantum Cellular Neural Network (MQ-CeNN) for Time Series Forecasting",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gedeonnkishi/mq-cenn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
