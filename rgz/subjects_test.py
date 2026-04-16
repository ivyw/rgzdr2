"""Tests for processing RGZ subjects."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from astropy.wcs import WCS
import numpy as np

import rgz.consensus
import rgz.constants
import rgz.subjects
import rgz.units as u

# Path to test directory.
_TEST_DIR = Path(os.path.dirname(__file__)) / "testdata/"

# Path to "cache" data.
_TEST_CACHE_DATA_PATH = _TEST_DIR / "first"

# Path to test (raw) subjects JSON.
_TEST_SUBJECTS_PATH = _TEST_DIR / "subjects.json"

# Path to test (processed) subjects JSON.
_TEST_SUBJECTS_PROCESSED_PATH = _TEST_DIR / "subjects_processed.json"


def make_dummy_wcs():
    # Make a dummy WCS 
    w = WCS(naxis=2)
    w.wcs.cdelt = [0.3, 0.3]
    w.wcs.crpix = [10, 15]
    w.wcs.crval = [2, 3]
    return w
    


class TestFindPointsInBox(unittest.TestCase):
    """Tests for rgz.subjects.find_points_in_box."""

    def test_simple(self):
        lower_ra = lower_dec = 0.0 * u.deg
        upper_ra = upper_dec = 1.0 * u.deg
        points = (
            np.array(
                [
                    [0.5, 0.5],
                    [0.2, 0.9],
                    [-0.1, 0.1],
                    [0.1, -0.1],
                    [-0.1, -0.1],
                ]
            )
            * u.deg
        )
        want = [0, 1]
        got = rgz.subjects.find_points_in_box(
            points, lower_ra, upper_ra, lower_dec, upper_dec
        )
        self.assertSetEqual(set(got), set(want))

    def test_ra_boundary(self):
        lower_dec = -1.0 * u.deg
        upper_dec = 1.0 * u.deg
        lower_ra = 359.9 * u.deg
        upper_ra = 0.1 * u.deg
        points = (
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                ]
            )
            * u.deg
        )
        want = [0]
        got = rgz.subjects.find_points_in_box(
            points, lower_ra, upper_ra, lower_dec, upper_dec
        )
        self.assertSetEqual(set(got), set(want))


class TestProcess(unittest.TestCase):
    """Tests for rgz.subjects.process."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self):
        """Tests behaviour consistency in processing subjects."""
        output_path = self.temp_dir_path / "out.json"
        rgz.subjects.process(_TEST_SUBJECTS_PATH, _TEST_CACHE_DATA_PATH, output_path)
        with open(output_path) as f:
            got = json.load(f)
        with open(_TEST_SUBJECTS_PROCESSED_PATH) as f:
            want = json.load(f)
        self.assertEqual(want, got)


# TODO(hzovaro): add test for serialisation/deserialisation - in particular that
# transforming a WCS from WCS object -> string -> object doesn't result in any 
# problematic differences 
# TODO(hzovaro): add test for transform_bbox_px_to_phys
class TestTransformCoordRadio(unittest.TestCase):
    """Tests for transform_coord_radio."""


    
    def test_all_pix2world(self):
        """Sanity check for WCS.all_pix2world."""
        # Create some dummy RA/Dec pairs and check that the common-sense 
        # transformation gives the same result as w.all_pix2world
        coords_phys = np.array([
            [1, 5], [-1, 10], [99, 100], [256, .3],
            [-99, 0.0003], [359., 179.], [1.3454, 63.0324],
        ])
        # NOTE: w.wcs.crpix is indexed from 1, so we have to subtract 1 as 
        # below.
        w = make_dummy_wcs()
        coords_px = (
            (coords_phys - w.wcs.crval) / w.wcs.cdelt
            + (w.wcs.crpix - 1)
        )
        want = coords_phys
        got = w.all_pix2world(coords_px, 0) 
        self.assertTrue(
            np.allclose(want, got)
        )


    def test_transform_coord_radio(self):
        w = make_dummy_wcs() 

        # Check for coords being out of bounds
        with self.assertRaises(ValueError):
            rgz.subjects.transform_coord_radio(np.array([-1, 0]), w)
            rgz.subjects.transform_coord_radio(np.array([132, 0]), w)
            rgz.subjects.transform_coord_radio(np.array([-1, 253]), w)

        # Check transformation has worked correctly
        for coords in [[0, 0], [1, 10], [99, 100], [131, 5]]:
            coords_unscaled_px = np.array([coords])
            coords_px = (
                coords_unscaled_px / rgz.constants.RADIO_MAX_PX * 100
            )
            coords_phys = (coords_px - (w.wcs.crpix - 1)) * w.wcs.cdelt + w.wcs.crval
            want = coords_phys
            got = rgz.subjects.transform_coord_radio(coords_unscaled_px[0],
                                                     wcs=w)
            self.assertTrue(np.allclose(
                want, np.array([[c.value for c in got]]))
            )


class TestTransformBboxPxToPhys(unittest.TestCase):
    """Tests for transform_bbox_px_to_phys."""
    # TODO(hzovaro): update once bboxes are refactored

    def test_transform(self):
        # TODO(hzovaro) test some more coords
        xmin = 1
        ymin = 1
        xmax = 2
        ymax = 2
        bbox = [
            xmin, ymin, xmax, ymax 
        ]
        w = make_dummy_wcs()
        want = np.concatenate(
            [
                rgz.subjects.transform_coord_radio(coord=np.array([xmin, 132 - ymax]), wcs=w),
                rgz.subjects.transform_coord_radio(coord=np.array([xmax, 132 - ymin]), wcs=w)
            ]
        )
        got = rgz.subjects.transform_bbox_px_to_phys(
            px_bbox=bbox,
            wcs=w
        )
        self.assertTrue(np.allclose(want, got))



if __name__ == "__main__":
    unittest.main()
