from setuptools import setup, find_packages

setup(
    name="aecf-ablation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "pandas",
    ],
)
