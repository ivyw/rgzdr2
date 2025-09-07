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
    fname = cache / f'{raw_subject["_id"]["$oid"]}.json'
    try:
        with open(fname) as f:
            response = json.load(f)
    except FileNotFoundError:
        response = requests.get(raw_subject["location"]["contours"]).json()
        with open(fname, "w") as f:
            json.dump(response, f)
    contours = response["contours"]
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
    
    return coords



def plot_classifications(
        subject: subjects.Subject,
        raw_subjects_path: Path,
        cache: Path,
        classifications: list[classifications.Classification] | None = None,
        ax: plt.Axes | None = None) -> plt.Axes:
    """TODO: write docstring"""

    # Find the raw subject belonging to this subject 
    # NOTE: I hate this!!!
    raw_subjects_jsons = []
    with open(raw_subjects_path, "r") as f:
        raw_subjects_rows = f.readlines()
        for row in raw_subjects_rows:
            raw_subjects_jsons.append(json.loads(row))
    raw_subject = [rs for rs in raw_subjects_jsons if rs["_id"]["$oid"] == subject.id][0]

    # Get the FIRST contours associated with this subject
    contour_coords = get_contours(
        raw_subject=raw_subject,
        cache=cache,
        px_coords=False,
    )

    # Get the WISE image associated with this subject 
    # TODO: subject.coords should probably return a SkyCoord...
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
        # TODO add WCS projection
        fig, ax = plt.subplots(subplot_kw=dict(projection=wcs_wise))
    else:
        # TODO destroy axis & add a new one with the same bbox with the correct 
        # projection
        fig = ax.get_figure()

    # Plot AllWISE image 
    # TODO use the same stretch as shown to citizen scientists
    ax.imshow(hdulist_wise[0].data)

    # Plot contours 
    ax.plot(*zip(*contour_coords), transform=ax.get_transform("fk5"),)

    # Add classifications 
    # TODO handle "NOSOURCE" etc etc
    for classification in classifications:
        for click in classification.coord_matches:
            coord_str, first_id = click
            click_coords = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
            ax.scatter(click_coords.ra.value, 
                    click_coords.dec.value,
                    marker="x",
                    c="r",
                    transform=ax.get_transform("fk5"),
                   )


    return ax 

