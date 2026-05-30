import json
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pathlib import Path

from rgz import subjects
from rgz import radio_islands
from rgz import constants
from rgz import classifications
from rgz.plot import classifications as plot_classifications
from rgz.plot import subject as plot_subject
from rgz import units as u


import sys

import matplotlib
import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

# Paths 
cache_path = Path("data") / "cache"
testdata_path = Path("rgz") / "testdata"
raw_subjects_path = testdata_path / "subjects.json"
processed_subjects_path = testdata_path / "subjects_processed.json"
processed_classifications_path = testdata_path / "classifications_processed.json"

"""
Sanity check the transformation between the weird RGZ coordinate system and 
physical coordinates.
"""

# with open(processed_subjects_path, "r") as f:
#     subject_instances = [subjects.Subject.from_json(s) for s in json.load(f)]

# make a radiosiland class instance 
cache = cache_path
sid = "52af7eb58c51f405a60012e6"
wcs = radio_islands.get_wcs(sid, cache)
bboxes = radio_islands.get_bboxes(sid, wcs=wcs, cache=cache)
contours_list = radio_islands.get_contours(sid, wcs=wcs, cache=cache)


sys.exit()


# Get bboxes and contours and plot them.
for s in subject_instances[:1]:

    # Plot the FIRST image 
    hdulist = radio_islands.load_first_image(s.id, cache=cache_path)
    im = hdulist[0].data 
    fig, ax = plt.subplots(subplot_kw={"projection": s.wcs})
    ax.imshow(im)

    # Get the raw bbox directly from the file 
    js = radio_islands.load_contour_data(s.id, cache_path)
    bboxes = []
    for contour in js["contours"]:
        bbox = contour[0]["bbox"]

        # Reset origin to zero, flip vertically, and scale.
        xmin_transformed = (bbox[0] - 1) * 100 / constants.RADIO_MAX_PX
        ymax_transformed = (constants.RADIO_MAX_PX - 1 - (bbox[1] - 1)) * 100 / constants.RADIO_MAX_PX
        xmax_transformed = (bbox[2] - 1) * 100 / constants.RADIO_MAX_PX
        ymin_transformed = (constants.RADIO_MAX_PX - 1 - (bbox[3] - 1)) * 100 / constants.RADIO_MAX_PX

        # Transform to RA/Dec.
        ra_min, dec_min = s.wcs.all_pix2world(np.array([[xmin_transformed, ymin_transformed]]), 0)[0] * u.deg
        ra_max, dec_max = s.wcs.all_pix2world(np.array([[xmax_transformed, ymax_transformed]]), 0)[0] * u.deg
        
        # Plot to check
        #TODO plot each line w/ a different colour to figure out dec min/max order.
        ax.plot([ra_min.value, ra_min.value, ra_max.value, ra_max.value, ra_min.value],
                [dec_min.value, dec_max.value, dec_max.value, dec_min.value, dec_min.value],
                transform=ax.get_transform("fk5"),
                color="r", ls="--", lw=3, label="Orignal - all_world2pix with origin = 0")

        # Repeat with contours 
        island_contours = []
        for island in js["contours"]:
            contours = []
            for contour in island:
                xs = [(coord["x"] - 1) * 100 / constants.RADIO_MAX_PX for coord in contour["arr"]]
                ys = [(constants.RADIO_MAX_PX - 1 - (coord["y"] - 1)) * 100 / constants.RADIO_MAX_PX for coord in contour["arr"]]
                # transform 
                coords = s.wcs.all_pix2world(np.vstack([xs, ys]).T, 0) * u.deg
                coords = [(x.value, y.value) for x, y in coords]
                contours.append(coords)
        island_contours.append(contours)

        for contour in island_contours[0]:
            ax.plot(
                *zip(*contour),
                color="green",
                path_effects=[
                    matplotlib.patheffects.withStroke(linewidth=2, foreground="red")
                ],
                transform=ax.get_transform("fk5"),
            )