"""Handles RGZ radio islands."""
from dataclasses import dataclass

from astropy.units import Quantity
from astropy.wcs import WCS

from rgz import units as u

def get_phys_bbox(self, bbox_px: BBox, wcs: ...) -> BBox_phys:
    """Returns a BBox_phys corresponding to a bbox_px according to a given World Coordinate System."""



@dataclass
class BBox:
    """Class for holding a bounding box."""
    xmin_px: float 
    ymin_px: float
    xmax_px: float 
    ymax_px: float


@dataclass
class BBox_phys:
    """Class for holding a bounding box."""
    xmin_phys: Quantity[u.deg]
    ymin_phys: Quantity[u.deg]
    xmax_phys: Quantity[u.deg]
    ymax_phys: Quantity[u.deg]


class RadioIsland:
    """A Radio Galaxy Zoo radio island.

    Radio islands are defined as contiguous radio sources in FIRST.
    
    Attributes: 

    TODO:
    Currently Subject doesn't have a WCS attribute. When the class is 
    constructed, it calls transform_bbox_px_to_phys for every bbox, which in turn uses
    transform_coord_radio TWICE to do the coordinate transform. Every time 
    it does this in needs to load the FITS file to get the WCS. Can improve 
    performance considerably by just storing the WCS object in Subject and 
    using this each time.

    *in my new branch it uses this in the constructor to get physical bboxs.
    In main it uses it to get FIRST ids. 

    process_subject ->
        for every bbox: 
            get_first_from_bbox
                --> transform_bbox_px_to_phys
                    --> 2x calls to transform_coord_radio
                        --> fetch_first_image_from_server_or_cache
    
    So 2 * # bboxes for every subject...

    """
    

    def __init__(self, 
                 bbox: BBox,
                 wcs: ...) -> None:
        self.bbox = bbox
        self.bbox_phys = get_phys_bbox() 


    