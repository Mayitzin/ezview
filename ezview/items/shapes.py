# -*- coding: utf-8 -*-
"""
Shapes
======

"""

import numpy as np

def ellipsoid(center: np.ndarray = None, axes: np.ndarray = None, num_points: int = 20) -> tuple:
    """
    Return the mesh of an ellipsoid.

    It builds the mesh of an ellipsoid given its center and axes lengths. The
    ellipsoid is defined by the equation:

    .. math::

        \\frac{x^2}{a^2} + \\frac{y^2}{b^2} + \\frac{z^2}{c^2} = 1

    The mesh is built using the parametric equations:

    .. math::

        x = a \\cos(u) \\sin(v) + x_0
        y = b \\sin(u) \\sin(v) + y_0
        z = c \\cos(v) + z_0

    The mesh is built using the numpy.mgrid function in the range
    :math:`u \\in [0, 2\\pi]` and :math:`v \\in [0, \\pi]`. The default number
    of points for the mesh is 20.

    The default values for the center and axes are [0, 0, 0] and [1, 1, 1],
    respectively, which corresponds to a unit sphere centered at the origin.

    Parameters
    ----------
    center : numpy.ndarray, optional
        3-element array with the ellipsoid's center. Default is [0, 0, 0].
    axes : numpy.ndarray, optional
        3-element array with the ellipsoid's main axes lengths. Default is
        [1, 1, 1].
    num_points : int, optional
        Number of points to use in the mesh. Default is 20.

    Returns
    -------
    tuple
        Tuple with the mesh of the ellipsoid in the form (x, y, z).

    Example
    -------
    >>> x, y, z = ezview.shapes.ellipsoid()
    >>> x.shape, y.shape, z.shape
    ((20, 20), (20, 20), (20, 20))
    >>> x, y, z = ezview.shapes.ellipsoid(center=[1, 2, 3], axes=[2, 3, 4], num_points=20)
    >>> x.shape, y.shape, z.shape
    ((20, 20), (20, 20), (20, 20))

    """
    if center is None:
        center = np.zeros(3)
    if axes is None:
        axes = np.ones(3)
    if not isinstance(center, (np.ndarray, list, tuple)):
        raise TypeError("Center must be a 3-element array.")
    if not isinstance(axes, (np.ndarray, list, tuple)):
        raise TypeError("Axes must be a 3-element array.")
    # Create ellipsoid mesh
    cx, cy, cz = center
    sx, sy, sz = axes
    u, v = np.mgrid[0:2*np.pi:complex(num_points), 0:np.pi:complex(num_points)]
    x = sx * np.cos(u)*np.sin(v) + cx
    y = sy * np.sin(u)*np.sin(v) + cy
    z = sz * np.cos(v) + cz
    return x, y, z

def frame(dcm: np.ndarray = None, position: np.ndarray = None, scale: float = 1.0) -> list:
    """
    Return the coordinates of an orthogonal frame.

    It returns the coordinates of an orthogonal frame given its direction
    cosine matrix (DCM) and position. The DCM is a 3-by-3 array with the
    frame's axes as columns.

    The position is a 3-element array with the frame's origin.

    The scale parameter is used to scale the frame's axes.

    If the DCM is None, it returns the identity matrix. If the position is
    None, it builds the frame from the origin.

    Parameters
    ----------
    dcm : numpy.ndarray, optional
        3-by-3 array with the frame's axes. Default is the identity matrix.
    position : numpy.ndarray, optional
        3-element array with the frame's origin. Default is [0, 0, 0].
    scale : float, optional
        Scale factor for the frame's axes. Default is 1.0.

    """
    if dcm is None:
        dcm = np.identity(3)
    if position is None:
        position = np.zeros(3)
    f_coords = []
    for column_index in range(3):
        axis_end = dcm[:, column_index]*scale + position
        f_coords.append(np.c_[position, axis_end])
    return f_coords