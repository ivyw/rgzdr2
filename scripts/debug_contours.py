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

from astropy.wcs import WCS
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


with open(processed_subjects_path, "r") as f:
    subject_instances = json.load(f)

# Get bboxes and contours and plot them.
zid = "ARG00031go"
s = [s for s in subject_instances if s["zid"] == zid][0]
id = s["id"]
wcs = WCS(s["wcs"])

# xmin, ymin, xmax, ymax
bboxes = radio_islands.get_bboxes(id, wcs=wcs, cache=cache_path)
contours = radio_islands.get_contours(id, wcs=wcs, cache=cache_path)

# Plot 
fig, ax = plt.subplots()
contour_colour = "white"
for island in contours:
    for contour in island:
        ax.plot(
            *zip(*contour),
            color=contour_colour,
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=2, foreground="red")
            ],
        )
for bbox in bboxes:
    xmin, ymin, xmax, ymax = bbox
    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        color="k"
    )