# -*- coding: utf-8 -*-
"""
EZView
======

Features
--------

- Fast visualization of data.
- Simplified calls and interfaces for the representation of data using
  matplotlib and Qt.

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided with the code, and
a loose standing reference guide built with Sphinx.

Viewing documentation using IPython
-----------------------------------
To see which functions are available in `ezview`, type ``ezview.<TAB>``
(where ``<TAB>`` refers to the TAB key). To view the docstring for a function,
use ``ezview.function_name?<ENTER>`` (to view the docstring) and
``ezview.function_name??<ENTER>`` (to view the source code).

"""

from .qplot import (
    QPlot,
    QPlotData,
    QPlot3D
)

MAJOR       = 0
MINOR       = 1
PATCH       = 0
PRE_RELEASE = ''

def get_version(short: bool = False) -> str:
    """Return the version number as a string."""
    if short or not PRE_RELEASE:
        return f"{MAJOR}.{MINOR}.{PATCH}"
    return f"{MAJOR}.{MINOR}.{PATCH}-{PRE_RELEASE}"
