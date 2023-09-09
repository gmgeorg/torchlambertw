from setuptools import setup, find_packages

setup(
    name="torchlambertw",
    version="0.0.1",
    url="https://github.com/gmgeorg/torchlambertw.git",
    author="Georg M. Goerg",
    author_email="im@gmge.org",
    description="torch implementation of Lambert W function and Lambert W x F distributions",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.0.0",
        "torch >= 2.0.1",
        "scipy>=1.0.0",
        "pytest>=6.1.1",
    ],
)
