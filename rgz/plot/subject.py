"""Plots an RGZ subject."""

from pathlib import Path

import astropy.io.ascii
import astropy.table
import matplotlib.axes
import matplotlib.patheffects
import numpy as np

from rgz import constants
from rgz import subjects
from rgz import units as u
from rgz.plot import contours as plot_contours
from rgz.plot import plotting
from rgz.plot import wise

_first_cat_cache = None


def plot_first(
    coords: tuple[float, float], cache: Path, ax: matplotlib.axes.Axes
) -> None:
    """Plots FIRST components as points with labels."""
    global _first_cat_cache
    if _first_cat_cache:
        table = _first_cat_cache
    else:
        table: astropy.table.Table = astropy.io.ascii.read(
            cache / constants.FIRST_CATALOGUE_NAME
        )  # pyright: ignore[reportAssignmentType]
        _first_cat_cache = table

    ras = np.asarray(table["RA_DEG"])
    decs = np.asarray(table["DE_DEG"])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    transform = ax.get_transform("fk5")  # pyright: ignore[reportCallIssue]

    # Curiously this function is no faster if we filter to only include points that are nearby.
    ax.scatter(
        ras * u.deg,
        decs * u.deg,
        transform=transform,
        color="w",
        marker="o",
        edgecolors="red",
    )

    centre_ra, centre_dec = coords
    im_width_deg = constants.IM_WIDTH_ARCMIN / 60
    for row in table:
        ra, dec = row["RA_DEG"], row["DE_DEG"]
        if (
            centre_ra - im_width_deg <= ra <= centre_ra + im_width_deg
            and centre_dec - im_width_deg <= dec <= centre_dec + im_width_deg
        ):
            ax.annotate(
                row["FIRST"],
                (ra, dec),
                xycoords=transform,
                xytext=(6, -3),
                color="white",
                textcoords="offset points",
                path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=2, foreground="red")
                ],
            )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_single_subject(
    subject: subjects.Subject, cache: Path, ax: matplotlib.axes.Axes | None
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
    wise.imshow(wise_image.data, ax)
    plot_contours.plot(island_contours, ax)
    # Overlay the FIRST objects in the area.
    plot_first(subject.coords, cache, ax)
    return ax
