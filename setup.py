# -*- coding: utf-8 -*-
"""
EZVIew is a package for fast visualization of data with Python using minimal
code calls.

EZView is compatible with Python 3.10 and newer.

"""

import sys
from setuptools import setup, find_packages
from .ezview import get_version

__version__ = get_version()

if sys.version_info < (3, 10):
    raise EnvironmentError("Python version >= 3.10 required. Python 3.11 is recommended")

metadata = dict(
    name='ezview',
    version=__version__,
    description='Data Visualization Tools.',
    long_description=__doc__,
    author='Mario Garcia',
    author_email='mario@myneeno.com',
    keywords="sensor plotting Qt data visualization",
    install_requires=['numpy',
                      'matplotlib',
                      'sip',
                      'pyqtgraph',
                      'PyQt5',
                      'PyOpenGL',
                      'PyOpenGL_accelerate'],
    packages=find_packages()
)

setup(**metadata)
