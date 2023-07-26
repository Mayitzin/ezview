"""
Module with the default colors to use.
"""

def hex_to_int(color: str) -> tuple:
    """Convert a hex color to a tuple of integers."""
    a = color.lstrip('#')
    return tuple(int(a[i:i+2], 16) for i in (0, 2, 4, 6))

def hex_to_float(color: str) -> tuple:
    """Convert a hex color to a tuple of floats."""
    a = color.lstrip('#')
    return tuple(int(a[i:i+2], 16)/255.0 for i in (0, 2, 4, 6))

COLORS = [
    "#FF0000FF", "#00AA00FF", "#0000FFFF", "#999933FF",
    "#FF8888FF", "#88AA88FF", "#8888FFFF", "#999955FF",
    "#660000FF", "#005500FF", "#000088FF", "#666600FF"]
COLORS_INTS = [hex_to_int(c) for c in COLORS]
COLORS_FLOATS = [hex_to_float(c) for c in COLORS]
