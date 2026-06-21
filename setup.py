"""Packaging configuration for the soccer betting modeling toolkit."""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="soccer-betting",
    version="0.1.0",
    description="Statistical & machine-learning models for soccer match probabilities and value betting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Soccer Betting Research Project",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "requests>=2.31",
        "PyYAML>=6.0",
        "matplotlib>=3.7",
    ],
    extras_require={
        "nn": ["torch>=2.0"],
        "catboost": ["catboost>=1.2"],
        "dev": ["pytest>=7.4"],
    },
    entry_points={
        "console_scripts": [
            "sb-collect=soccer_betting.cli:collect_main",
            "sb-train=soccer_betting.cli:train_main",
            "sb-backtest=soccer_betting.cli:backtest_main",
            "sb-value=soccer_betting.cli:value_main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
