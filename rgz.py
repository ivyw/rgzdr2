"""Utilities for interacting with RGZ raw data."""

import collections
import enum
import itertools
import json
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
warnings.simplefilter('ignore', astropy.wcs.FITSFixedWarning)
warnings.simplefilter('ignore', urllib3.connectionpool.InsecureRequestWarning)

from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as skcoord
from astroquery.image_cutouts.first import First

from astropy.io import fits
from astropy.wcs import WCS

IR_MAX_PX = 424
RADIO_MAX_PX = 132
IM_WIDTH_ARCMIN = 3

def get_bboxes(subject: dict[str, ...]) -> tuple[tuple[float, float, float, float], ...]:
    """Fetches the bboxes of a subject from RGZ, caching locally."""
    fname = f'first/{subject["_id"]["$oid"]}.json'
    try:
        with open(fname) as f:
            js = json.load(f)
    except FileNotFoundError:
        url = subject['location']['contours']
        response = requests.get(url)
        if not response.ok:
            raise RuntimeError('Error:', response.status_code)
        js = response.json()
        assert abs(js['width'] - 132) <= 1
        with open(fname, 'w') as f:
            json.dump(js, f)
    bboxes = []
    for contour in js['contours']:
        assert contour[0]['k'] == 0
        bboxes.append(tuple([round(c, 1) for c in contour[0]['bbox']]))
    return tuple(bboxes)

def download_first_image(raw_subject: dict[str, ...]) -> fits.HDUList:
    """Downloads a FIRST image from the FIRST server."""
    coord = raw_subject['coords']
    coord = skcoord.SkyCoord(ra=coord[0], dec=coord[1], unit='deg')
    fname = f'first/{raw_subject["_id"]["$oid"]}.fits'
    try:
        return fits.open(fname)
    except FileNotFoundError:
        im = First.get_images(coord, image_size=3 * u.arcmin)
        im.writeto(fname)
        return im

def get_classifications(path='radio_classifications.json') -> dict[str, ...]:
    """Yields classifications from RGZ."""
    with open(path, encoding='utf-8') as f:
        # each row is a JSON document
        for row in f:
            js = json.loads(row)
            # if js['subject_ids'][0]['$oid'] not in all_subjects:
            #     # sometimes not true??
            #     continue
            for anno in js['annotations']:
                if 'radio' in anno:
                    break
            else:
                # no annotations?
                continue
            yield js

@attr.s
class Classification:
    cid = attr.ib()
    zid = attr.ib()
    matches = attr.ib()
    username = attr.ib()
    notes = attr.ib()

@attr.s
class Subject:
    id = attr.ib()
    zid = attr.ib()
    coords = attr.ib()
    bboxes = attr.ib()

def transform_coord_ir(coord: tuple[float, float], raw_subject: dict[str, ...]=None, wcs:...=None):
    if not raw_subject and not wcs:
        raise ValueError()
    if raw_subject:
        assert not wcs
        wcs = get_wcs(raw_subject)
    # coord in 424x424 -> 424x424
    coord = coord * 100 / 424
    # flip y axis?
    coord[1] = 100 - coord[1]
    c = wcs.all_pix2world([coord], 0)[0] * u.deg
    return c

def process_classification(classification: dict[str, ...], subject: Subject, wcs: ..., defer_ir_lookup=False) -> Classification:
    """Converts a raw classification into a Classification."""
    cid = classification['_id']['$oid']
    zid = classification['subjects'][0]['zooniverse_id']
    if zid != subject.zid:
        raise ValueError('Mismatched subjects.')
    matches = []  # (wise, first)
    notes = []
    for anno in classification['annotations']:
        if 'radio' not in anno:
            continue
        boxes = set()
        if anno['radio'] == 'No Contours':
            # ?????? ignore this
            continue
        for radio in anno['radio'].values():
            box = tuple(round(float(radio[corner]), 1) for corner in ['xmax', 'ymax', 'xmin', 'ymin'])
            boxes.add(box)
        
        if anno['ir'] == 'No Sources':
            ir = 'NOSOURCE'
        else:
            if len(anno['ir']) != 1:
                notes.append('MULTISOURCE')
            ir_coord = anno['ir']['0']['x'], anno['ir']['0']['y']
            ir_coord = np.array([float(i) for i in ir_coord])
            ir_coord = transform_coord_ir(ir_coord, wcs=wcs)
            ir_coord = skcoord.SkyCoord(ra=ir_coord[0].value, dec=ir_coord[1].value,
                                        unit=(ir_coord[0].unit, ir_coord[0].unit),
                                        frame='icrs')
            if not defer_ir_lookup:
                # query the IR
                q = Vizier.query_region(ir_coord,
                                        radius=5 * u.arcsec,
                                        catalog=['II/328/allwise'])
                try:
                    ir = q[0][0]['AllWISE']
                except IndexError:
                    ir = f'NOMATCH_J{ir_coord.to_string("hmsdms", sep="").replace(" ", "")}'
            else:
                ir = ir_coord.to_string()
        matches.append((ir, [c for b in boxes for c in subject.bboxes[b]]))
    return Classification(cid=cid, zid=zid, matches=matches, username=classification.get('user_name', 'NO_USER_NAME'), notes=notes)

def plot_contours(raw_subject: dict[str, ...], ax=None, bbox_plot_kwargs=None, px_coords=False, px_scaling=100 / RADIO_MAX_PX):
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
        response = requests.get(raw_subject['location']['contours']).json()
        with open(fname, 'w') as f:
            json.dump(response, f)
    contours = response['contours']
    for contour in contours:
        contour = contour[0]
        xs = [a['x'] for a in contour['arr']]
        ys = [a['y'] for a in contour['arr']]
        coords = np.stack([xs, ys]).T
        if not px_coords:
            coords = [transform_coord_radio(c, raw_subject) for c in coords]
            coords = [(a.value, b.value) for a, b in coords]
        else:
            coords = [(c[0] * px_scaling, 100 - c[1] * px_scaling) for c in coords]
        plt.plot(*zip(*coords))
#     plt.xlim(plt.xlim()[::-1])
    
    bboxes = get_bboxes(raw_subject)
    bboxes = [transform_bbox(bbox, raw_subject) for bbox in bboxes]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.value
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c='k', **bbox_plot_kwargs)

def transform_coord_radio(coord: tuple[int, int], raw_subject: dict[str, ...]) -> u.Quantity:
    """Transforms a radio image pixel coordinate.

    Note that this uses the WCS of the subject image, and can be slow!

    TODO: Speed this up by avoiding the image reload whenever possible, e.g. by passing in the image.
    """
    im = download_first_image(raw_subject)
    header = im[0].header
    # WCS.dropaxis doesn't seem to work on these images
    # drop these: CTYPE3 CRVAL3 CDELT3 CRPIX3 CROTA3
    for key in ['CTYPE', 'CRVAL', 'CDELT', 'CRPIX', 'CROTA']:
        for i in [3, 4]:
            del header[key + str(i)]
    wcs = WCS(header)
    # coord in 132x132 -> 100x100
    coord = coord * 100 / 132
    # flip y axis
    c = wcs.all_pix2world([coord], 0)[0] * u.deg
    return c

def transform_bbox(bbox, raw_subject):
    bbox_ = bbox
    bbox = np.array(bbox)
    bbox = np.concatenate([transform_coord_radio(bbox[:2], raw_subject),
                           transform_coord_radio(bbox[2:], raw_subject)])
    return bbox
    

def get_first_from_bbox(bbox, raw_subject, verbose=False):
    # TODO: might need to flip horizontally or even vertically...
    bbox = transform_bbox(bbox, raw_subject)
    # find the centre
    centre = (bbox[::2].mean(), bbox[1::2].mean())
    # and the width, height
    width = abs(bbox[2] - bbox[0]).to(u.arcsec)
    height = abs(bbox[3] - bbox[1]).to(u.arcsec)
    
    # round widths and heights up to nearest arcsec plus two
    width = np.ceil(width.to(u.arcsec)) + 2 * u.arcsec
    height = np.ceil(height.to(u.arcsec)) + 2 * u.arcsec
    
    if verbose:
        print('get_first_from_bbox:', centre, width, height)
    skc = skcoord.SkyCoord(ra=centre[0].value, dec=centre[1].value,
                           unit=(centre[0].unit, centre[0].unit),
                           frame='icrs')
    # Now we can do a VizieR query.
    q = Vizier.query_region(skc,
                            width=width,
                            height=height,
                            catalog=['VIII/92/first14'])
    try:
        return list(q[0]['FIRST'])
    except IndexError:
        return [f'NOFIRST_J{skc.to_string("hmsdms", sep="").replace(" ", "")}']

def plot_raw_subject(raw_subject: dict[str, ...], scaling: int=1):
    f = download_first_image(raw_subject)
    wcs = WCS(f[0].header)
    ax = plt.subplot(projection=wcs, slices=('x', 'y', 0, 0))
    ax.imshow(f[0].data)

    bboxes = get_bboxes(raw_subject)
    for bbox in bboxes:
        bbox = np.array(bbox)
        bbox *= 100 / 132
        x1, y1, x2, y2 = bbox
        y2 = 100 - y2
        y1 = 100 - y1
        ax.plot([x1, x1, x2, x2, x1], [y2, y1, y1, y2, y2], c='white')

    # transformed bboxes
    bboxes = [transform_bbox(bbox, raw_subject) for bbox in bboxes]
    for bbox in bboxes:
        c1 = skcoord.SkyCoord([bbox[:2]], frame='icrs', unit='deg')
        c2 = skcoord.SkyCoord([bbox[2:]], frame='icrs')
        bbox = np.array(bbox)
        x1, y1, x2, y2 = bbox
        ax.plot([x1, x1, x2, x2, x1], [y2, y1, y1, y2, y2], c='cyan', transform=ax.get_transform('world'))