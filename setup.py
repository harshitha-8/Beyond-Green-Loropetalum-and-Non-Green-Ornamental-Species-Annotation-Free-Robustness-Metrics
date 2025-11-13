#!/usr/bin/env python
"""
Setup script for Beyond Green: Annotation-Free Robustness Metrics
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Annotation-Free Robustness Metrics for Non-Green Ornamental Plant Detection"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="beyond-green-robustness-metrics",
    version="0.1.0",
    author="Harshitha M",
    author_email="harshitha.m@utdallas.edu",
    description="Annotation-Free Robustness Metrics for Non-Green Ornamental Plant Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshitha-8/Beyond-Green-Loropetalum-and-Non-Green-Ornamental-Species-Annotation-Free-Robustness-Metrics",
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.1.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "yolov8": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "ultralytics>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "beyond-green-process=experiments.cvpr_optimized_processing:main",
            "beyond-green-compare=experiments.classical_vs_yolov8_comparison:main",
            "beyond-green-metrics=experiments.loropetalum_metrics_generation:main",
            "beyond-green-evaluate=experiments.strategy_evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    keywords=[
        "computer vision",
        "plant detection",
        "robustness metrics",
        "annotation-free",
        "UAV imagery",
        "ornamental plants",
        "Loropetalum",
    ],
    project_urls={
        "Bug Reports": "https://github.com/harshitha-8/Beyond-Green-Loropetalum-and-Non-Green-Ornamental-Species-Annotation-Free-Robustness-Metrics/issues",
        "Source": "https://github.com/harshitha-8/Beyond-Green-Loropetalum-and-Non-Green-Ornamental-Species-Annotation-Free-Robustness-Metrics",
        "Paper": "https://openreview.net/forum?id=SD6FZaEJAH",
    },
)
