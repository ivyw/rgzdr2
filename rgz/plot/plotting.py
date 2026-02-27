"""Plotting utilities used by all plotting code."""

import astropy.wcs
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

def maybe_create_axes(
        ax: matplotlib.axes.Axes | None,
        wcs: astropy.wcs.WCS) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    if ax is None:
        return plt.subplots(subplot_kw=dict(projection=wcs))
    # Replace existing axes with ones with the correct projection
    fig: matplotlib.figure.Figure = ax.get_figure() # pyright: ignore[reportAssignmentType]
    bbox = ax.get_position()
    ax.remove()
    ax = fig.add_axes(rect=bbox, projection=wcs)
    return fig, ax
