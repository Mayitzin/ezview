# -*- coding: utf-8 -*-
"""
EZVIew is a package for fast visualization of data with Python using minimal
code calls.

EZView is compatible with Python 3.10 and newer.

"""

import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 10):
    raise EnvironmentError("Python version >= 3.10 required.")

metadata = dict(
    name='ezview',
    version='0.1.0',
    description="Light visualization toolkit for motion data.",
    packages=find_packages(),
    long_description=__doc__,
    author='Mario Garcia',
    author_email='mario@myneeno.com',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="sensor plotting Qt data visualization",
    install_requires=['numpy',
                      'matplotlib>=3.6',
                      'sip',
                      'pyqtgraph',
                      'PyQt5',
                      'PyOpenGL',
                      'PyOpenGL_accelerate'],
    python_requires=">=3.10",
)

setup(**metadata)
