#!/usr/bin/env python3
"""
FractionalTorch: Exact Rational Arithmetic for Numerically Stable Neural Network Training

A PyTorch extension that replaces floating-point arithmetic with exact rational 
number representations, achieving perfect numerical reproducibility and improved 
training stability.
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    init_file = os.path.join(os.path.dirname(__file__), 'fractionaltorch', '__init__.py')
    if os.path.exists(init_file):
        with open(init_file, 'r') as f:
            content = f.read()
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
            if version_match:
                return version_match.group(1)
    return "0.1.0"  # fallback version

# Read long description from README
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def get_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'fractions',
    ]

# Development requirements
dev_requirements = [
    'pytest>=6.0.0',
    'pytest-cov>=2.10.0',
    'black>=21.0.0',
    'flake8>=3.8.0',
    'mypy>=0.800',
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    'jupyter>=1.0.0',
    'matplotlib>=3.3.0',
    'seaborn>=0.11.0',
]

# Documentation requirements
docs_requirements = [
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    'sphinxcontrib-napoleon>=0.7',
    'myst-parser>=0.15.0',
]

# Benchmark requirements
benchmark_requirements = [
    'matplotlib>=3.3.0',
    'seaborn>=0.11.0',
    'pandas>=1.3.0',
    'scikit-learn>=1.0.0',
    'tensorboard>=2.7.0',
]

setup(
    name="fractionaltorch",
    version=get_version(),
    author="[Your Name]",
    author_email="[your.email@example.com]",
    description="Exact rational arithmetic for numerically stable neural network training",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/FractonicMind/FractionalTorch",
    project_urls={
        "Bug Reports": "https://github.com/FractonicMind/FractionalTorch/issues",
        "Source": "https://github.com/FractonicMind/FractionalTorch",
        "Documentation": "https://fractionaltorch.readthedocs.io/",
        "Paper": "https://arxiv.org/abs/2025.XXXXX",
        "Blog Post": "https://medium.com/@leogouk/why-your-neural-networks-fail-and-how-i-fixed-it-562376bc88ad",
    },
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*', 'benchmarks*']),
    classifiers=[
        # Development Status
        "Development Status :: 3 - Alpha",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # Topic
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating Systems
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        
        # Environment
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: Console",
        
        # Natural Language
        "Natural Language :: English",
    ],
    keywords=[
        "pytorch", "neural networks", "machine learning", "deep learning",
        "numerical stability", "exact arithmetic", "fractional arithmetic",
        "reproducible research", "precision", "rational numbers"
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "benchmarks": benchmark_requirements,
        "all": dev_requirements + docs_requirements + benchmark_requirements,
    },
    entry_points={
        "console_scripts": [
            "fractionaltorch-benchmark=fractionaltorch.cli.benchmark:main",
            "fractionaltorch-convert=fractionaltorch.cli.convert:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fractionaltorch": [
            "py.typed",  # PEP 561 type information
            "data/*.json",
            "configs/*.yaml",
        ],
    },
    zip_safe=False,  # Required for mypy to find type information
    
    # Additional metadata for PyPI
    platforms=["any"],
    license="MIT",
    
    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=6.0.0",
        "pytest-cov>=2.10.0",
    ],
    
    # Options for bdist_wheel
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal because we may have C extensions
        }
    },
    
    # Command classes for custom setup commands
    cmdclass={
        # Can add custom commands here if needed
    },
)
