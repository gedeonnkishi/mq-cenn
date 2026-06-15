from setuptools import setup

setup(
    name="mq-cenn",
    version="1.0.0",
    py_modules=["mq_cenn"],
    install_requires=[
        "numpy>=1.22.0",
        "torch>=2.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.4.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
    ],
    author="Gedeon Nkishi",
    description="MQ-CeNN framework for time-series forecasting",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gedeonnkishi/mq-cenn",
    python_requires=">=3.8",
)
