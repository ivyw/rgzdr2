import json
from pathlib import Path
import requests

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt 
import numpy as np

from rgz import rgz
from rgz import constants
from rgz import classifications 
from rgz import cutouts
from rgz import subjects



def get_contours(
    raw_subject: rgz.JSON,
    px_coords=False,
    px_scaling=100 / constants.RADIO_MAX_PX,
    cache: Path = Path("first"),
) -> list[tuple]:
    """Returns the contours of a raw subject."""
    # TODO I think there is something wrong with this
    fname = cache / f'{raw_subject["_id"]["$oid"]}.json'
    try:
        with open(fname) as f:
            response = json.load(f)
    except FileNotFoundError:
        response = requests.get(raw_subject["location"]["contours"]).json()
        with open(fname, "w") as f:
            json.dump(response, f)
    contours = response["contours"]
    coords_list = []
    for contour in contours:
        contour = contour[0]
        xs = [a["x"] for a in contour["arr"]]
        ys = [a["y"] for a in contour["arr"]]
        coords = np.stack([xs, ys]).T
        if not px_coords:
            coords = [
                subjects.transform_coord_radio(c, raw_subject, cache=cache) for c in coords
            ]
            coords = [(a.value, b.value) for a, b in coords]
        else:
            coords = [(c[0] * px_scaling, 100 - c[1] * px_scaling) for c in coords]
        coords_list.append(coords)
    return coords_list


def get_raw_subject(subject: subjects.Subject,
                    raw_subjects_path: Path,
                    ) -> rgz.JSON:
    """Returns the "raw" subject associated with a Subject."""
    # TODO read directly from FIRST directory instead of taking from the "master" JSON file
    # the contours are stored in testdata/first/<oid>.json
    # NOTE: I hate this!!!
    # NOTE $oid seems to be unique 
    raw_subjects_jsons = []
    with open(raw_subjects_path, "r") as f:
        raw_subjects_rows = f.readlines()
        for row in raw_subjects_rows:
            raw_subjects_jsons.append(json.loads(row))
    raw_subjects_list = [rs for rs in raw_subjects_jsons if rs["_id"]["$oid"] == subject.id]
    if len(raw_subjects_list) > 1:
        raise AssertionError(f"I found {len(raw_subjects_list)} raw subjects associated with Subject {subject.id}!")
    return raw_subjects_list[0]


def get_first_coords_from_id(first_id: str) -> SkyCoord:
    """Returns a SkyCoord object corresponding to FIRST component names."""
    assert (first_id.startswith("FIRST") or first_id.startswith("NOFIRST"))
    # TODO should probably use regex for this
    first_coord_nospaces_str = first_id.split("_J")[1]
    sign_str = "+" if "+" in first_coord_nospaces_str else "-"
    ra_nospaces_str, dec_nospaces_str = first_coord_nospaces_str.split(sign_str)
    ra_str = ra_nospaces_str[:2] + " " + ra_nospaces_str[2:4] + " " + ra_nospaces_str[4:]
    dec_str = dec_nospaces_str[:2] + " " + dec_nospaces_str[2:4] + " " + dec_nospaces_str[4:]
    dec_str = sign_str + dec_str
    first_coords_str = ra_str + " " + dec_str
    return SkyCoord(first_coords_str, unit=(u.hourangle, u.deg))


def plot_single_classification(
        subject: subjects.Subject,
        raw_subjects_path: Path,
        cache: Path,
        classification: classifications.Classification,
        ax: plt.Axes | None = None) -> plt.Axes:
    """Plot a single classification.
    
    NOTE: the FIRST radio images linked in the raw_subjects JSON appear to be 
    upside down. See here: https://github.com/ivyw/rgzdr2/issues/9#issuecomment-3315480856 
    """
    # TODO input checks - check that the subject matches the classification

    # Get the raw subject so we can get the contours
    raw_subject = get_raw_subject(subject=subject, 
                                  raw_subjects_path=raw_subjects_path)

    # Get the FIRST contours associated with this subject
    # TODO these don't have any associated labels or FIRST ids...
    contour_coords_list = get_contours(
        raw_subject=raw_subject,
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
        # TODO destroy axis & add a new one with the same bbox with the correct 
        # projection
        fig = ax.get_figure()

    # Plot AllWISE image 
    ax.imshow(hdulist_wise[0].data, cmap="gist_heat", vmax=6, vmin=2)

    # Plot contours 
    # TODO can annotate these w/ FIRST component names
    contour_colour = "y"
    for contour_coords in contour_coords_list:
        ax.plot(*zip(*contour_coords), 
                transform=ax.get_transform("fk5"),
                color=contour_colour,)

    # Add classifications 
    # TODO some classifications are "empty" - i.e. coord_matches is []
    # what do these mean?
    # TODO does it make sense to plot multiple classifications on the same plot?
    # esp. since we should probably plot the contours corresponding to different
    # matches in different colours - e.g. if the subject has 3 radio islands and
    # a user has associated 2 of them with 1 IR source and the other radio 
    # island with another IR source
    # We could group together classifications that have made the SAME associations - 
    # i.e. show the IR source clicks from all users that have chosen the same 
    # subset of radio islands. 
    # TODO can we link first_id below to the contour_coords above?
    # TODO do we want to make a fancy GUI where you can flick through the 
    # different classifications for the same subject? 
    colours = plt.rcParams["axes.prop_cycle"].by_key()['color']
    if len(classification.coord_matches) == 0:
        ax.text(s="coord_matches list is empty!",
                x=0.5, y=0.5, transform=ax.transAxes, va="center", ha="center",
                color="white",
                )
    else:
        for cc, click in enumerate(classification.coord_matches):
            coord_str, first_ids = click
            # Label for this IR source match shown in the legen
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
                    transform=ax.get_transform("fk5"),
                    c=colours[cc],
                    label=f"{source_label} - FIRST component {ff + 1}",
                )

            # Plot the location of the IR host
            if coord_str == "NOSOURCE":
                continue
            click_coords = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
            ax.scatter(click_coords.ra.value, 
                    click_coords.dec.value,
                    marker="x",
                    label=f"{source_label} - IR host",
                    transform=ax.get_transform("fk5"),
                    c=colours[cc],
                    )
        
        # Decorations 
        ax.legend(loc="center left", bbox_to_anchor=[1.05, 0.5])
    ax.set_title(f"Zooniverse ID {subject.zid} ({subject.id})")


    return ax 

