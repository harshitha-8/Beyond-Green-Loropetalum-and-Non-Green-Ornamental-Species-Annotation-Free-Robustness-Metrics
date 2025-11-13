from setuptools import setup, find_packages

setup(
    name="loropetalum-annotation-free-metrics",
    version="1.0.0",
    author="Anonymous CVPR Submission",
    description="Annotation-Free Robustness Metrics for Dense Plant Counting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/harshitha-8/Beyond-Green-Loropetalum-and-Non-Green-Ornamental-Species-Annotation-Free-Robustness-Metrics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
    ],
)
