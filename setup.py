"""Setup file."""

from setuptools import find_packages, setup

import re

_VERSION_FILE = "torchlambertw/_version.py"
verstrline = open(_VERSION_FILE, "rt").read()
_VERSION = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(_VERSION, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (_VERSION_FILE,))


setup(
    name="torchlambertw",
    version=verstr,
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
