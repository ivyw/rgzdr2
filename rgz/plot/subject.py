"""Plots an RGZ subject."""

from pathlib import Path

import matplotlib.axes

from rgz import subjects
from rgz.plot import contours as plot_contours
from rgz.plot import plotting
from rgz.plot import wise


def plot_single_subject(
    subject: subjects.Subject,
    cache: Path,
    ax: matplotlib.axes.Axes | None
) -> matplotlib.axes.Axes:
    """Plots a single subject."""
    island_contours = plot_contours.get_contours(
        subject=subject,
        cache=cache,
        px_coords=False,
    )
    # We need the WISE image to get the WCS.
    wise_image = wise.get_wise_image(subject.coords)
    fig, ax = plotting.maybe_create_axes(ax, wcs=wise_image.wcs)