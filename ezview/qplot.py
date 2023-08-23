# -*- coding: utf-8 -*-
"""
Plotting Tools with Qt
======================

Classes for fast plotting of 2D and 3D data using PyQtGraph.

"""
# PSL
from collections.abc import Callable

# Third-Party Libraries
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Valid shaders. See: https://github.com/pyqtgraph/pyqtgraph/blob/master/pyqtgraph/opengl/shaders.py
VALID_SHADERS = ['balloon', 'viewNormalColor', 'normalColor', 'shaded', 'edgeHilight', 'heightColor', 'pointSprite']
COLORS_FLOATS = [(1., 0., 0., 1.), (0., 1., 0., 1.), (0., 0., 1., 1.),
               (0.5, 0., 0., 1.), (0., 0.5, 0., 1.), (0., 0., 0.5, 1.)]
COLORS_INT = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255),
              (127, 0, 0, 255), (0, 127, 0, 255), (0, 0, 127, 255)]
DEG2RAD = np.pi/180.0

class QPlot:
    """
    :class:`QPlot` creates a Qt Application managing GUI application's control
    flow and main settings.

    Parameters
    ----------
    size : 2-tuple of ints, default: (800, 600)
        Figure dimension ``(width, height)`` in pixels.
    antialias : bool, default: True
        Enable antialiasing to draw lines with smooth edges at the cost of
        reduced performance.

    Attributes
    ----------
    app : pyqtgraph.Qt.QtGui.QApplication
        Qt Application
    winsize : tuple, default: (800, 600)
        Window Size
    """
    def __init__(self, **kw):
        self.app = pg.mkQApp("Data Visualization.")
        self.winsize = kw.get('size', (800, 600))
        pg.setConfigOptions(antialias=kw.get('antialias', True))
        self.timer = None

    def launch_app(self):
        """
        Enters the main event loop and waits until exit() is called, then
        returns the value that was set to exit(), which is 0 if exit() is
        called via quit().

        Generally, no user interaction can take place before calling this
        method. It is necessary to call this function to start event handling.
        The main event loop receives events from the window system and
        dispatches these to the application widgets.

        References
        ----------
        .. [Qt5-QApplication] https://doc.qt.io/qt-5/qapplication.html
        """
        self.app.instance().exec_()

    def add_timed_action(self, slot: Callable, timeout: int = 0):
        """
        Create a timer, and connect to ``slot`` with a ``timeout`` in msec.

        Parameters
        ----------
        slot : Callable
            Slot to connect to. Defines the action to take when the timeout `n`
            is reached.
        timeout : int, default: 0
            Time in msec for the timeout. The timer restarts at `n`
            milliseconds and the slot `f` is then called.

        References
        ----------
        .. [Qt5-QTimer] https://doc.qt.io/qt-5/qtimer.html
        .. [PyQt4-QTimer] https://www.riverbankcomputing.com/static/Docs/PyQt4/qtimer.html
        """
        self.timer = pg.Qt.QtCore.QTimer()
        self.timer.timeout.connect(slot)
        self.timer.start(timeout)


class QPlotData(QPlot):
    """
    The :class:`QPlotData` contains most of the figure elements of a plot along
    X-axis (time dimension)
    """
    def __init__(self, data=None, **kw):
        """
        Parameters
        ----------
        data : numpy.ndarray, default: None
            M-by-N NumPy array with the data to plot, where `M` is the number
            of elements to plot, and `N` is the number of samples per element.
        size : 2-tuple of ints, default: (800, 600)
            Figure dimension ``(width, height)`` in pixels.
        antialias : bool, default: True
            Enable antialiasing to draw lines with smooth edges at the cost of
            reduced performance.

        Attributes
        ----------
        app : pyqtgraph.Qt.QtGui.QApplication
            Qt Application
        winsize : tuple, default: (800, 600)
            Window Size
        win : GraphicsWindow
            Graphics Layout Window
        p : PlotItem
            Plot Item
        lines : list
            Lines to show
        num_samples : int
            Number of samples to show.
        data : numpy.ndarray, default: None
            M-by-N NumPy array with the data to plot, where `M` is the number
            of elements to plot, and `N` is the number of samples per element.
            Its shape must coincide with (len(lines), num_samples)
        """
        super().__init__()
        self.data = data
        self.lines = []
        self.num_samples = 0 if self.data is None else self.data.shape[-1]
        self.win = self.add_window(**kw)
        self.p = self.add_plot(**kw)
        if self.data is not None:
            if self.data.ndim > 2:
                print(f"[ERROR] Expected data with one or two dimensions. Got {self.data.ndim} dimensions.")
            elif self.data.ndim > 1:
                self.lines = [self.add_line(r, color=COLORS_INT[i]) for i, r in enumerate(self.data)]
            else:
                self.lines = [self.add_line(self.data, color=COLORS_INT[0])]

    def add_window(self, **kw):
        """
        Create a Widget Layout in the current Plotting Window.

        Parameters
        ----------
        title : string, default: 'data over time'
            Set the title of the Window.
        """
        title = kw.get('title', "data over time")
        # Create Graphics Layout Widget
        win = pg.GraphicsLayoutWidget(show=True)
        win.setWindowTitle(title)
        win.resize(*self.winsize)
        return win

    def add_plot(self, **kw):
        """
        Add a plot item (canvas) to the Widget Layout.

        Plot item that is added to current graphics layout. Implements axes,
        titles, and interactive viewbox. The plot item also provides some basic
        analysis functionality that may be accessed from the context menu.

        Parameters
        ----------
        subtitle : string, default: None
            Subtitle of the plot item.
        aspect_ratio : float, default: None
            Aspect ratio of the plot when rescaled. If is None
        samples : int, default: 200
            Maximum number of samples to show per line.
        xrange : 2-tuple of floats, default (0, samples)
            Initial range along the X-axis.
        yrange : 2-tuple of floats, default (0.0, 1.0)
            Initial range along the Y-axis.
        grid : string, default: ''
            Show grid for desired axis. Options are ``'x'``, ``'y'`` and
            ``'xy'``. If none is given (``''``), no grid is shown.

        Returns
        -------
        p : pyqtgraph.PlotItem
            Created plot item (canvas.)

        References
        ----------
        .. [pyqtg-plotitem] http://www.pyqtgraph.org/documentation/graphicsItems/plotitem.html
        """
        subtitle = kw.get('subtitle')
        grid = kw.get('grid', '').lower()
        self.num_samples = kw.get('samples', self.num_samples)
        # Create and add a plot canvas to the widget layout
        p = self.win.addPlot()
        if subtitle is not None:
            p.setTitle(subtitle)
        # Set Ranges
        p.setXRange(*kw.get('xrange', (0.0, self.num_samples)))
        if 'yrange' in kw:
            p.setYRange(*kw.get('yrange', (0.0, 1.0)))
        else:
            p.enableAutoRange()
        # Grid Item
        p.showGrid(x='x' in grid, y='y' in grid)
        return p

    def add_line(self, data=None, color='r'):
        """
        Add a stroke in the scene.

        Parameters
        ----------
        data : array, default: None.
            Array with the coordinates of the stroke. If None is given, nothing
            is plotted, but an empty instance is still created and returned.
        color : str or tuple, default: 'w'.
            String of desired color or its corresponding tuple.

        Returns
        -------
        stroke : PlotDataItem
            Item added to the Scene.
        """
        if data is None:
            return self.p.plot()
        data = np.copy(data)
        if self.data is None:
            self.data = data.flatten()
        if data.ndim > 1:
            data = data.flatten()
        return self.p.plot(data, pen=color)

    def update_data(self, data):
        """
        Update data array
        """
        if self.data.ndim > 1:
            self.data = np.roll(self.data, -1, axis=1)
            self.data[:, -1] = data
        else:
            self.data = np.roll(self.data, -1)
            self.data[-1] = data


class QPlot3D(QPlot):
    """
    The :class:`QPlot3D` contains most of the figure elements of a
    three-dimensional scene.
    """
    def __init__(self, data=None, **kw):
        """
        Parameters
        ----------
        data : numpy.ndarray
            N-by-3 numpy.ndarray of three-dimensional coordinates of each sample.
        antialias : bool, default: True
            Enable antialiasing to draw lines with smooth edges at the cost of
            reduced performance.
        title : str, optional. Default: "3D Visualization"
            Title of the Window.
        distance : float, optional. Default: "distance"
            Distance between camera center and origin.
        antialias : bool, optional. Default: False
            Enable antialiasing to causes lines to be drawn with smooth edges
            at the cost of reduced performance.
        scattered : bool, optional. Default: False
            Display data as point cloud.

        Attributes
        ----------
        win : GLViewWidget
            Graphics Layout View Widget
        p : PlotItem
            Plot Item
        """
        super().__init__()
        self.win = self.add_window(**kw)
        self.add_grid('z', -1)
        self.split = kw.get("split")
        self.scattered = kw.get("scattered", False)
        self.data = data
        self.lines = []
        self.frames = kw.get('frames')
        self.frames_positions = kw.get('frames_positions')
        if self.data is not None:
            if self.data.shape[1] != 3 or self.data.ndim != 2:
                raise ValueError(f"Input data must be of shape (N, 3). Got {self.data.shape}")
            else:
                if self.split is not None:
                    try:
                        for s in self.split:
                            if self.scattered:
                                self.add_scatter(self.data[s[0]:s[1], :])
                            else:
                                self.add_line(self.data[s[0]:s[1], :])
                    except ValueError as e:
                        print(e, "Check your trace spans")
                else:
                    if self.scattered:
                        self.add_scatter(self.data)
                    else:
                        self.lines.append(self.add_line(self.data))
        if self.frames is not None:
            for idx, frame in enumerate(self.frames):
                frame_pos = self.frames_positions[idx] if self.frames_positions is not None else None
                self.add_frame(frame, position=frame_pos, scale=0.25)
        self.launch_app()

    def add_window(self, **kw):
        """
        Create a basic GL Widget to display 3D data.

        Parameters
        ----------
        title : string, default: '3D Visualization'
            Title of the created window.
        dist : float, default: 4.0
            Distance of camera to the origin.

        Returns
        -------
        win : pyqtgraph.opengl.GLViewWidget
            Graphics Layout View Widget
        """
        title = kw.get('title', "3D Visualization")
        dist = kw.get('distance', 4.0)
        # Create GLViewWidget
        win = gl.GLViewWidget()
        win.setWindowTitle(title)
        win.resize(*self.winsize)
        win.show()
        win.setCameraPosition(distance=dist)
        return win

    def add_item(self, item):
        """
        Add an item to the layout and place it in the scene.

        Parameters
        ----------
        item : PlotItem
            Plot Item to add.
        """
        self.win.addItem(item)

    def add_line(self, data=None, **kw):
        """
        Add a line to the scene.

        It defaults to a red sphere of radius equal to 1 and centered in the
        origin.

        Parameters
        ----------
        data : numpy.ndarray
            N-by-3 numpy.ndarray of three-dimensional coordinates of each sample.
        color : list of floats, optional. Default: [1.0, 0.0, 0.0, 1.0]
            Color of sphere in float values between 0.0 and 1.0 with the RGBA
            format.
        width : float, default: 1.0
            Width of the line.

        Returns
        -------
        points : pyqtgraph.opengl.GLLinePlotItem
            Created line plot item.
        """
        if data is None:
            data = np.zeros((1, 3))
        if data.shape[1] != 3 or data.ndim != 2:
            raise ValueError(f"Input data must be of shape (N, 3). Got {data.shape}")
        color = kw.get('color', COLORS_FLOATS[3])
        width = kw.get('width', 1.0)
        antialias = kw.get('antialias', False)
        line = gl.GLLinePlotItem(pos=data, color=color, antialias=antialias, width=width)
        self.win.addItem(line)
        return line

    def add_scatter(self, data=None, color=None, size=0.01, **kw):
        """
        Add a point cloud to the scene.

        Parameters
        ----------
        data : numpy.ndarray
            N-by-3 numpy.ndarray of three-dimensional coordinates of each sample.
        color : list of floats, optional. Default: [1.0, 0.0, 0.0, 1.0]
            Color of sphere in float values between 0.0 and 1.0 with the RGBA
            format.
        size : float
            Size of each scattered point. Default is 0.01

        Returns
        -------
        points : pyqtgraph.opengl.GLScatterPlotItem
            Created scatter plot item.
        """
        c = color if color is not None else COLORS_FLOATS[0]
        d = data if data is not None else np.zeros(3)
        alpha = kw.get('alpha')
        if alpha:
            c[-1] = alpha
        points = gl.GLScatterPlotItem(pos=d, size=size, color=c, pxMode=False)
        self.win.addItem(points)
        return points

    def add_spheroid(self, center=None, scale=None, color=None, **kw):
        """
        Add a 3D wired sphere item to the scene.

        It defaults to a red sphere of radius equal to 1 and centered in the
        origin.

        Parameters
        ----------
        center : list of floats, default: [0.0, 0.0, 0.0]
            Position of spheroid's center.
        scale : list of floats, default: [1.0, 1.0, 1.0]
            Scale factor along the three axis of sphere.
        color : list of floats, default: [1.0, 0.0, 0.0, 1.0]
            Color of sphere in float values between 0.0 and 1.0 with the RGBA
            format. Defaults to a red color with full alpha chanel.
        alpha : float, optional. Default: 1.0
            If alpha is not None, it forces the alpha value to 1.0, except if
            color is None, which always maps to [1.0, 0.0, 0.0, 1.0]
        shader : string, optional. Default: 'balloon'
            Set the shader used when rendering faces in the mesh. Valid shaders
            are: 'balloon', 'viewNormalColor', 'normalColor', 'shaded',
            'edgeHilight', 'heightColor' and 'pointSprite'

        Returns
        -------
        sphere : pyqtgraph.opengl.GLMeshItem
            Created sphere item.
        """
        md = gl.MeshData.sphere(rows=20, cols=20)
        c = color if color else list(COLORS_FLOATS[0])
        s = scale if scale else [1.0, 1.0, 1.0]
        p = center if center else [0.0, 0.0, 0.0]
        # Read extra parameters, if any
        alpha = kw.get('alpha')
        if alpha:
            c[-1] = alpha
        shader = kw.get('shader', 'balloon')
        if shader not in VALID_SHADERS:
            shader = 'balloon'
        # Build sphere
        sphere = gl.GLMeshItem(meshdata=md, smooth=True, color=c, shader=shader, glOptions='additive')
        sphere.scale(*s)
        sphere.translate(*p)
        self.win.addItem(sphere)
        return sphere

    def add_frame(self, dcm=None, position=None, colors=None, scale=1.0):
        """
        Add a 3D frame in the scene. Each axis is visualized with RGB color
        sequence, by default.

        The method returns a list of the three axes of the frame, so that they
        can be called and used from an external application.

        Parameters
        ----------
        dcm : array
            3-by-3 Direction Cosine Matrix. Defaults to identity matrix
        position : array, default: [0.0, 0.0, 0.0]
            three-dimensional coordinates of frame's origin
        colors : list
            3 colors to use for each axis of the frame. Each color must be
            specified with a tuple of 4 floats.
        scale : float, default: 1.0
            Length of each frame axis

        Returns
        -------
        frame : list
            List of each axis line plot item.
        """
        if dcm is None:
            dcm = np.identity(3)
        if position is None:
            position = np.zeros(3)
        if colors is None:
            colors = COLORS_FLOATS[:3]
        frame = []
        for column_index, axis_color in enumerate(colors):
            axis_end = np.array([
                dcm[0, column_index]*scale + position[0],
                dcm[1, column_index]*scale + position[1],
                dcm[2, column_index]*scale + position[2]
                ])
            f_coords = np.vstack((position, axis_end))
            frame.append(self.add_line(f_coords, color=axis_color, antialias=True))
        return frame

    def add_grid(self, axis='z', shift=0):
        """
        Add a wire-grame grid to the Scene.

        Parameters
        ----------
        axis : string, default: 'z'
            Normal axis of the grid plane.
        shift : float, default: 0
            Displacement of the grid along its normal vector.

        Returns
        -------
        grid : pyqtgraph.opengl.GLGridItem
            Created grid item.
        """
        if axis.lower() not in ['x', 'y', 'z']:
            return None
        grid = gl.GLGridItem()
        if axis == 'x':
            grid.rotate(90, 0, 1, 0)
            if shift != 0:
                grid.translate(shift, 0, 0)
        if axis == 'y':
            grid.rotate(90, 1, 0, 0)
            if shift != 0:
                grid.translate(0, shift, 0)
        if axis == 'z' and shift != 0:
            grid.translate(0, 0, shift)
        self.win.addItem(grid)
        return grid
