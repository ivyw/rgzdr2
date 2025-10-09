import json
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np

from rgz import constants
from rgz import classifications
from rgz import cutouts
from rgz import subjects


def get_contours(
    subject: subjects.Subject,
    px_coords: bool = False,
    px_scaling: float = 100 / constants.RADIO_MAX_PX,
    cache: Path = Path("first"),
) -> list[tuple]:
    """Returns the contours of a raw subject."""
    fname = cache / f"{subject.id}.json"
    try:
        with open(fname) as f:
            response = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"contour data for subject with ID {subject.id}" f"not found!"
        )
    contours = response["contours"]
    coords_list = []
    for contour in contours:
        contour = contour[0]
        xs = [a["x"] for a in contour["arr"]]
        ys = [a["y"] for a in contour["arr"]]
        coords = np.stack([xs, ys]).T
        if not px_coords:
            coords = [
                subjects.transform_coord_radio(
                    coord=c, subject=subject, raw_subject=None, cache=cache
                )
                for c in coords
            ]
            coords = [(a.value, b.value) for a, b in coords]
        else:
            coords = [(c[0] * px_scaling, 100 - c[1] * px_scaling) for c in coords]
        coords_list.append(coords)
    return coords_list


def get_first_coords_from_id(first_id: str) -> SkyCoord:
    """Returns a SkyCoord object with coordinates extracted from a FIRST ID."""
    assert first_id.startswith("FIRST") or first_id.startswith("NOFIRST")
    # TODO should probably use regex for this
    first_coord_nospaces_str = first_id.split("_J")[1]
    sign_str = "+" if "+" in first_coord_nospaces_str else "-"
    ra_nospaces_str, dec_nospaces_str = first_coord_nospaces_str.split(sign_str)
    ra_str = (
        ra_nospaces_str[:2] + " " + ra_nospaces_str[2:4] + " " + ra_nospaces_str[4:]
    )
    dec_str = (
        dec_nospaces_str[:2] + " " + dec_nospaces_str[2:4] + " " + dec_nospaces_str[4:]
    )
    dec_str = sign_str + dec_str
    first_coords_str = ra_str + " " + dec_str
    return SkyCoord(first_coords_str, unit=(u.hourangle, u.deg))


def plot_single_classification(
    subject: subjects.Subject,
    classification: classifications.Classification,
    cache: Path,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a single classification.

    This function plots the WISE image and radio contours associated with a
    subject, overlaid with markers indicating the classification made by a
    citizen scientist. The stretch of the WISE image is intended to be identical
    to that shown to citizen scientists during the classification process.

    IR host galaxy click locations are represented by crosses, and the
    corresponding radio sources are indicated with circular markers.
    Individual host galaxy-radio source associations are indicated by markers
    of the same colour. If the input classification has

    Args:
        subject: the subject to plot
        classification: the classification to plot. The Zooniverse ID must match
            that of the subject.
        cache: path to raw subject data.
        ax: Matplotlib axes (optional). Note that axes passed to this function
            are first destroyed and re-created with the WCS of the WISE image
            because there is no way to modify the axis projection after it has
            been created. The bbox of the original axes is used to create the
            new axes, so it will remain in the same place in the figure.

    Returns:
        Matplotlib axes containing the plot.

    Raises:
        ValueError if the subject Zooniverse ID does not match that of the
        input classification.
    """
    # Check that the subject matches the classification
    if subject.zid != classification.zid:
        raise ValueError(
            f"subject with Zooniverse id {subject.zid} does not "
            "match that of input classification with id "
            f"{classification.zid}!"
        )

    # Get the FIRST contours associated with this subject
    contour_coords_list = get_contours(
        subject=subject,
        cache=cache,
        px_coords=False,
    )

    # Get the WISE image associated with this subject
    ra, dec = subject.coords
    coords_wise = SkyCoord(ra, dec, unit="deg")
    hdulist_wise = cutouts.get_allwise_cutout(
        coords=coords_wise,
        size=constants.IM_WIDTH_ARCMIN * u.arcmin,
        save_fits=False,
    )
    wcs_wise = WCS(hdulist_wise[0].header)

    # Create figure and axes
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=wcs_wise))
    else:
        # Replace existing axes with ones with the correct projection
        fig = ax.get_figure()
        bbox = ax.get_position()
        ax.remove()
        ax = fig.add_axes(rect=bbox, projection=wcs_wise)

    ax.set_title(f"Zooniverse ID {subject.zid}\n(subject ID: {subject.id})")

    # Plot AllWISE image. The colour map and limits have been calibrated to
    # match what was shown to citizen scientists as closely as possible
    ax.imshow(hdulist_wise[0].data, cmap="gist_heat", vmax=6, vmin=2)

    # Plot contours
    # TODO annotate these with FIRST IDs, and colour the contours to indicate
    # the source rather than overplotting a scatter marker.
    contour_colour = "white"
    for contour_coords in contour_coords_list:
        ax.plot(
            *zip(*contour_coords),
            transform=ax.get_transform("fk5"),
            color=contour_colour,
        )

    # Add classification
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # If the coord_matches list is empty, display a warning on the plot and
    # return.
    if len(classification.coord_matches) == 0:
        ax.text(
            s="coord_matches list is empty!",
            x=0.5,
            y=0.5,
            transform=ax.transAxes,
            va="center",
            ha="center",
            color="black",
        )
        return ax

    # Otherwise, plot each match.
    for cc, click in enumerate(classification.coord_matches):
        coord_str, first_ids = click
        # Label for this IR source match shown in the legend
        if coord_str == "NOSOURCE":
            source_label = "No source"
        else:
            source_label = f"Source {cc + 1}"

        # Plot the locations of the FIRST IDs
        for ff, first_id in enumerate(first_ids):
            first_component_coords = get_first_coords_from_id(first_id)
            ax.scatter(
                first_component_coords.ra.value,
                first_component_coords.dec.value,
                marker="o",
                edgecolors="k",
                transform=ax.get_transform("fk5"),
                c=colours[cc],
                label=f"{source_label} - FIRST component {ff + 1}",
            )

        # Plot the location of the IR host
        if coord_str == "NOSOURCE":
            continue
        click_coords = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
        ax.scatter(
            click_coords.ra.value,
            click_coords.dec.value,
            marker="x",
            label=f"{source_label} - IR host",
            transform=ax.get_transform("fk5"),
            c=colours[cc],
        )

    # Decorations
    ax.legend(loc="center left", bbox_to_anchor=[1.05, 0.5])

    return ax
