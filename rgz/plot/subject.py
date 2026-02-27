"""Plots an RGZ subject."""

from pathlib import Path

import matplotlib.axes

from rgz import subjects
from rgz.plot import contours as plot_contours
from rgz.plot import plotting


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
    fig, ax = maybe_create_axes