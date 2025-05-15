"""Utilities for interacting with RGZ raw data."""

import collections
import enum
import itertools
import json
import logging
import os
import time
from pathlib import Path

import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

import astropy.wcs
import warnings
import urllib3

import rgz.subjects
from rgz.subjects import Subject

warnings.simplefilter("ignore", astropy.wcs.FITSFixedWarning)
warnings.simplefilter("ignore", urllib3.connectionpool.InsecureRequestWarning)

from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as skcoord
from astroquery.image_cutouts.first import First

from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

def plot_contours(
    raw_subject: dict[str, ...],
    ax=None,
    bbox_plot_kwargs=None,
    px_coords=False,
    px_scaling=100 / rgz.subjects.RADIO_MAX_PX,
):
    """Plots the contours of a raw subject."""
    if not ax:
        ax = plt.gca()
    if not bbox_plot_kwargs:
        bbox_plot_kwargs = {}

    fname = f'first/{raw_subject["_id"]["$oid"]}.json'
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
                rgz.subjects.transform_coord_radio(c, raw_subject) for c in coords
            ]
            coords = [(a.value, b.value) for a, b in coords]
        else:
            coords = [(c[0] * px_scaling, 100 - c[1] * px_scaling) for c in coords]
        plt.plot(*zip(*coords))
    #     plt.xlim(plt.xlim()[::-1])

    bboxes = rgz.subjects.get_bboxes(raw_subject)
    bboxes = [rgz.subjects.transform_bbox(bbox, raw_subject) for bbox in bboxes]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.value
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c="k", **bbox_plot_kwargs)


def plot_raw_subject(raw_subject: dict[str, ...], scaling: int = 1):
    f = rgz.subjects.download_first_image(raw_subject)
    wcs = WCS(f[0].header)
    ax = plt.subplot(projection=wcs, slices=("x", "y", 0, 0))
    ax.imshow(f[0].data)

    bboxes = rgz.subjects.get_bboxes(raw_subject)
    for bbox in bboxes:
        bbox = np.array(bbox)
        bbox *= 100 / 132
        x1, y1, x2, y2 = bbox
        y2 = 100 - y2
        y1 = 100 - y1
        ax.plot([x1, x1, x2, x2, x1], [y2, y1, y1, y2, y2], c="white")

    # transformed bboxes
    bboxes = [rgz.subjects.transform_bbox(bbox, raw_subject) for bbox in bboxes]
    for bbox in bboxes:
        c1 = skcoord.SkyCoord([bbox[:2]], frame="icrs", unit="deg")
        c2 = skcoord.SkyCoord([bbox[2:]], frame="icrs")
        bbox = np.array(bbox)
        x1, y1, x2, y2 = bbox
        ax.plot(
            [x1, x1, x2, x2, x1],
            [y2, y1, y1, y2, y2],
            c="cyan",
            transform=ax.get_transform("world"),
        )
