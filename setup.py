"""
Setup script for AECF - Adaptive Ensemble CLIP Fusion
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aecf",
    version="1.0.0",
    description="Adaptive Ensemble CLIP Fusion - Production-ready multi-modal learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AECF Team",
    author_email="aecf@example.com",
    url="https://github.com/your-org/aecf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="multi-modal, deep-learning, pytorch, fusion, computer-vision, nlp",
    project_urls={
        "Bug Reports": "https://github.com/your-org/aecf/issues",
        "Source": "https://github.com/your-org/aecf",
        "Documentation": "https://aecf.readthedocs.io/",
    },
)
