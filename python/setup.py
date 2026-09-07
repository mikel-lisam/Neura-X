# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Setup script for Neura-X Python package.
#
# ==========================================================

from setuptools import setup, find_packages

setup(
    name="neura-x",
    version="1.0.0",
    description="Intelligence Without Limits. CPU-first universal AI framework.",
    long_description=open("../README.md").read() if __import__("os").path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    author="Edusei Mikel Lisamba",
    author_email="lisambamikel@gmail.com",
    url="https://github.com/mikeledusei/Neura-X",
    license="Neura-X Dual License",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "full": [
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "transformers>=4.30.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)