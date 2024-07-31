# -*- coding: utf-8 -*-
"""
Plotting Data Tools
===================

This module uses matplotlib as its plotting engine to produce visualizations
for documentation and journal quality.

"""

import numpy as np
import matplotlib.pyplot as plt
from .colors import COLORS
from .items import ellipsoid
from .items import frame

def add_ellipsoid(ax, params: list | dict, num_points: int = 20, color = 'k', lw = 0.5, **kwargs) -> None:
    """
    Add a ellipsoid to an existing 3D plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        3D axis where the ellipsoid will be added.
    params : list or dict.
        List or dictionary with the parameters to draw an ellipsoid. If a list
        is given, it must be of the form [[a, b, c], [x, y, z]], where a, b, c
        are the coordinates of the ellipsoid's center, and x, y, z are the
        ellipsoid's main axes lengths. If a dictionary is given, it must be of
        the form {'center': [a, b, c], 'axes': [x, y, z]}
    num_points : int, optional
        Number of points, per axis, to use in the mesh. Default is 20.
    color : str, optional
        Color of the ellipsoid. Default is 'k'.
    lw : float, optional
        Line width of the ellipsoid. Default is 0.5.

    """
    if isinstance(params, (list, tuple, np.ndarray)):
        center, axes = params
    elif isinstance(params, dict):
        center = params.get("center", np.zeros(3))
        axes = params.get("axes", np.ones(3))
    else:
        raise TypeError("Unknown type for 'sphere'. Try a list or a dict.")
    # Extract only the expected parameters from kwargs
    expected_params = {'num_points': num_points, 'color': color, 'lw': lw}
    for key in expected_params:
        if key in kwargs:
            expected_params[key] = kwargs[key]
    x, y, z = ellipsoid(center=center, axes=axes, num_points=expected_params['num_points'])   # Ellipsoid mesh
    ax.plot_wireframe(x, y, z, color=expected_params['color'], lw=expected_params['lw'])

def add_frame(ax, dcm, position = None, color: str | list = None, scale: float = 1.0, lw: float = 1.0) -> None:
    """
    Add a frame to an existing 3D plot.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axis where the frame will be added.
    frame : numpy.ndarray
        3-by-3 array with the frame's axes. Each row is a vector.
    color : str or list of strings, optional
        Color of the frame. Default is None, which iterates over RGB.
    lw : float, optional
        Line width of the frame. Default is 1.0.

    """
    if not hasattr(ax, 'plot'):
        raise TypeError("The given axis is not a 3D plot item.")
    colors = ([color]*3 if isinstance(color, str) else color) if color is not None else COLORS[:3]
    frame_coords = frame(dcm, position, scale)
    for axis in frame_coords:
        ax.plot(*axis, color=colors.pop(0), lw=lw)

def add_items(ax, **kwargs) -> None:
    """
    Add items to an existing 3D plot.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axis where the items will be added.
    kwargs : dict
        Dictionary with the items to be added. The keys are the items' types,
        and the values are the items' data and parameters.

    """
    if 'scatter' in kwargs:
        for k, data in kwargs['scatter'].items():
            if isinstance(data, (list, tuple, np.ndarray)):
                if isinstance(data, np.ndarray):
                    data = data.T
                ax.scatter(*data)
            elif isinstance(data, dict):
                data = data.copy()
                points = data.pop('data')
                ax.scatter(*points.T, **data)
            else:
                raise TypeError(f"Unknown type for 'scatter': {type(data)}. Try a list or a dict.")
    if 'lines' in kwargs:
        for k, data in kwargs['lines'].items():
            if isinstance(data, (list, tuple, np.ndarray)):
                if isinstance(data, np.ndarray):
                    data = data.T
                ax.plot(*data)
            elif isinstance(data, dict):
                data = data.copy()
                lines = data.pop('data')
                ax.plot(*lines.T, **data)
            else:
                raise TypeError(f"Unknown type for 'lines': {type(data)}. Try a list or a dict.")
    if 'frames' in kwargs:
        for k, v in kwargs['frames'].items():
            add_frame(ax, v['attitude'], v['position'])
    if 'ellipsoids' in kwargs:
        for k, v in kwargs['ellipsoids'].items():
            add_ellipsoid(ax, v, **v)

def plot_data(*data, **kw):
    """
    Plot data with custom formatting.

    Given data is plotted in time domain. It locks any current process until
    plotting window is closed.

    Multiple arrays of data can be given as first parameters. It creates
    subplots for each array automatically and stacks them vertically, so that
    their X-axis is shared.

    Parameters
    ----------
    data : array
        Arrays with the contents of data to plot. They could be 1- (single line)
        or 2-dimensional.
    title : int or str
        Window title as number or label.
    subtitles : list
        List of strings of the titles of each subplot.
    labels : list
        List of labels that will be displayed in each subplot's legend.
    xlabels : list
        List of strings of the labels of each subplot's X-axis.
    ylabels : list
        List of strings of the labels of each subplot's Y-axis.
    yscales : str
        List of strings of the scales of each subplot's Y-axis. It supports
        matlabs defaults values: "linear", "log", "symlog" and "logit"

    Examples
    --------
    >>> data = [2., 3., 4., 5.]
    >>> plot_data(data)
    >>> data_2 = [4., 5., 6., 7.]
    >>> plot_data(data, data_2)
    >>> plot_data(data, data_2, subtitles=["data 1", "data 2"])

    Arrays with different dimensions can be plotted together.

    >>> data1 = np.random.random((1000, 3))     # 1000-by-3 array
    >>> data2 = np.random.random((500, 4))      # 500-by-4 array
    >>> plot(data1, data2, labels=['array 1', 'array 2'])
    """
    title = kw.get("title")
    subtitles = kw.get("subtitles")
    labels = kw.get("labels")
    xlabels = kw.get("xlabels")
    ylabels = kw.get("ylabels")
    yscales = kw.get("yscales")
    index = kw.get("index")
    indices = kw.get("indices")
    shades_spans = kw.get("shadeTouch")
    num_subplots = len(data)        # Number of given arrays
    # Create figure with vertically stacked subplots
    fig, axs = plt.subplots(
        num_subplots,
        1,
        num=title,
        squeeze=False,
        sharex=kw.get('sharex', "indices" not in kw),
        sharey=kw.get('sharey', False)
        )
    for i, array in enumerate(data):
        array = np.copy(array)
        if array.ndim > 2:
            raise ValueError(f"Data array {i} has more than 2 dimensions.")
        if array.ndim < 2:
            # Plot a single line in the subplot (1-dimensional array)
            label = labels[i][0] if labels else None
            index = index if index is not None else np.arange(array.shape[0])
            axs[i, 0].plot(index, array, color=COLORS[0], lw=0.5, ls='-', label=label)
        else:
            # Plot multiple lines in the subplot (2-dimensional array)
            array_sz = array.shape
            if array_sz[0] > array_sz[1]:
                # Transpose array if it has more rows than columns
                array = array.T
            index = indices[i] if indices is not None else np.arange(array_sz.shape[1])
            for j, row in enumerate(array):
                label = None
                if labels:
                    if len(labels[i]) == len(array):
                        label = labels[i][j]
                axs[i, 0].plot(index, row, color=COLORS[j], lw=0.5, ls='-', label=label)
        axs[i, 0].grid(axis='y')
        if subtitles:
            axs[i, 0].set_title(subtitles[i])
        if xlabels:
            axs[i, 0].set_xlabel(xlabels[i])
        if ylabels:
            axs[i, 0].set_ylabel(ylabels[i])
        if yscales:
            axs[i, 0].set_yscale(yscales[i])
        if shades_spans is not None:
            # Add shaded areas corresponding to TOUCHING areas
            try:
                if isinstance(shades_spans, (list, np.ndarray)):
                    current_spans = shades_spans[i] if np.copy(shades_spans).ndim > 2 else shades_spans
                    for s in current_spans:
                        axs[i, 0].axvspan(s[0], s[1], color='gray', alpha=0.1)
                elif isinstance(shades_spans, dict):
                    # Add shades AND their corresponding labels
                    for k, v in shades_spans.items():
                        span = [v['start'], v['stop']]
                        axs[i, 0].axvspan(span[0], span[1], color='gray', alpha=0.1)
                        axs[i, 0].text(int(np.mean(span)), max(array), k, ha='center')
            except:
                print("No spans were given")
        if labels:
            if len(labels[i]) > 0:
                axs[i, 0].legend(loc='lower right')
    fig.tight_layout()
    plt.show()

def plot3(**kwargs) -> None | tuple:
    """
    Plot 3-dimensional data in a cartesian coordinate system.

    Parameters
    ----------
    show : bool, optional
        Show the plot after creating it. Default is True. Otherwise, the plot
        Figure and Axes objects are returned.

    """
    # Build the plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Add items
    add_items(ax, **kwargs)

    # Set properties of plot
    plt.tight_layout()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')      # Added in matplotlib 3.6

    # Show or return the plot
    if not kwargs.get('show', True):
        return fig, ax
    plt.show()

def plot_kinematics(data, ref_data: np.ndarray = None, tip: np.ndarray = None, split: np.ndarray = None) -> None:
    """
    Plot 3 main kinematic properties: position, velocity and acceleration.

    ``data`` must be an array, whose columns represent groups of position,
    velocity and acceleration, respectively::

        data = [x, x', x'']

    where each group can be a one-, two- or three-dimensional array with N 
    samples.

    Parameters
    ----------
    data : numpy.ndarray
        N-by-M array of data, where M is a multiple of 3.
    ref_data : numpy.ndarray, default: None
        N-by-M reference data. It must have the same size as ``data`` array.
    tip : numpy.ndarray
        N-by-3 array of tip data.
    split : numpy.ndarray
        N-by-2 array of stroke spans.
    """
    _, N = data.shape
    if ref_data is not None:
        if ref_data.shape!=data.shape:
            raise ValueError("Reference data does not match the main data")
    dims = N//3
    #### Plot each kinematic property (left pane)
    properties = {"Acceleration": "m / s^2", "Velocity": "m / s", "Position": "m"}
    fig = plt.figure()
    for j, (k, v) in enumerate(properties.items()):
        ax = fig.add_subplot(3, 2, 2*j+1)
        plt.title(k)
        for i in range(dims):
            plt.plot(data[:, i+(2-j)*dims], COLORS[i], ls='-', lw=1.0 if ref_data is not None else 0.5)
            if ref_data is not None:
                plt.plot(ref_data[:, i+(2-j)*dims], COLORS[i], ls='--', alpha=0.3)
        if split is not None:
            for span in split:
                plt.axvspan(span[0], span[1], color='red', alpha=0.1)
        plt.ylabel(v)
        plt.grid(True, which='major', axis='y')
    #### Motion Reconstruction (right pane)
    if dims < 3:
        ax = fig.add_subplot(1, 2, 2)
        if dims < 2:
            plt.plot(data[:, 0], 'g-')
            if ref_data is not None:
                plt.plot(ref_data.flatten(), 'b--')
        else:
            plt.plot(data[:, 0], data[:, 1], 'g-')
            if ref_data is not None:
                plt.plot(ref_data[:, 0], ref_data[:, 1], 'b--')
            plt.axis('equal')
            plt.grid(True, which='major', axis='both')
    else:
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot3D(data[:, 0], data[:, 1], data[:, 2], COLORS[0], lw=0.7)
        if ref_data is not None:
            ax.plot3D(ref_data[:, 0], ref_data[:, 1], ref_data[:, 2], COLORS[1], alpha=0.5)
        if tip is not None:
            if split is not None:
                for span in split:
                    x, y, z = tip[span[0]:span[1]].T
                    ax.plot3D(x, y, z, COLORS[2], alpha=0.75)
            else:
                ax.plot3D(tip[:, 0], tip[:, 1], tip[:, 2], COLORS[2], alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')      # Added in matplotlib 3.6
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run directly.")
