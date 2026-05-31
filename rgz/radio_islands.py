"""Handles RGZ radio islands."""
import logging
from typing import Self

import numpy as np

from rgz import bboxes
from rgz import first
from rgz import rgz
from rgz import units as u


logger = logging.getLogger(__name__)


class RadioIsland:
    """A Radio Galaxy Zoo radio island.

    Radio islands are defined as contiguous radio sources in FIRST.
    
    Attributes: 
    - physical-unit bounding box 
    - contours 

    Methods:
    - get_firsts


    """

    def __init__(self, 
                 rgzbbox: bboxes.RGZBBox,
                 bbox: bboxes.BBox,
                 contours: list[list[tuple]], # TODO what is this?
                 first_tree: first.FIRSTTree | None = None,
                 firsts: list[first.FIRSTID] | None = None,
                 ) -> None:
        """Initialise a RadioIsland instance."""
        # Input validation
        if (first_tree is None) and (firsts is None):
            raise ValueError("first_tree must be specified if firsts is None!")
        if (first_tree is not None) and (firsts is not None):
            raise ValueError("Only one of first_tree and firsts may be specified!")
        if (firsts is not None):
            if len(firsts) == 0:
                raise ValueError("firsts must not be empty!")

        # Initialise instance attributes
        self.rgzbbox = rgzbbox
        self.bbox = bbox
        self.contours = contours  # TODO(hzovaro): input check this
        if firsts is not None:
            self.firsts = firsts
        else:
            self.firsts = self.get_firsts(first_tree)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RadioIsland):
            return ValueError("other must be a BBox!")
        return (self.bbox == other.bbox) &\
            (self.rgzbbox == other.rgzbbox) &\
            (self.contours == other.contours) &\
            (self.firsts == other.firsts)
    
    def get_firsts(self, first_tree: first.FIRSTTree,
    ) -> list[first.FIRSTID]:
        """Finds FIRST components within a bounding box."""
        # TODO(MatthewJA): Also use the contours to ensure that they really are within the boxes.

        padded_bbox = bboxes.BBox(
            ra_min=np.ceil(self.bbox.ra_min.to(u.arcsec)) - 1 * u.arcsec,
            ra_max=np.ceil(self.bbox.ra_max.to(u.arcsec)) + 1 * u.arcsec,
            dec_min=np.ceil(self.bbox.dec_min.to(u.arcsec)) - 1 * u.arcsec,
            dec_max=np.ceil(self.bbox.dec_max.to(u.arcsec)) + 1 * u.arcsec,
        )
        matching_indices = padded_bbox.get_points_in_box(first_tree[0] * u.deg)
        if not matching_indices:
            coord_str = rgz.coord_to_string(self.bbox.centre)
            ra_hh, ra_mm, ra_ss, dec_dd, dec_mm, dec_ss = [float(s) for s in coord_str.split(" ")]
            coord_str = f"NOFIRST_J{ra_hh:02.0f}{ra_mm:02.0f}{ra_ss:0.1f}{dec_dd:+02.0f}{dec_mm:02.0f}{dec_ss:02.0f}"
            return [f'NOFIRST_J{coord_str.replace(" ", "")}']

        names = []
        for index in matching_indices:
            names.append("FIRST_" + first_tree[1][index])
        return sorted(names)

    def to_json(self) -> rgz.JSON:
        """Converts a RadioIsland into a JSON-compatible dictionary."""
        return {
            "bbox": self.bbox.to_json(), 
            "rgzbbox": self.rgzbbox,
            "contours": self.contours,
            "firsts": self.firsts,
        }
    
    @classmethod
    def from_json(cls, radioisland: rgz.JSON) -> Self:
        """Reads a RadioIsland from JSON."""
        return cls(
            bbox=bboxes.BBox.from_json(radioisland["bbox"]),
            rgzbbox=radioisland["rgzbbox"],
            contours=radioisland["contours"],
            firsts=radioisland["firsts"],
        )