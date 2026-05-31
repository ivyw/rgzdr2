"""Tests for processing the RGZ radioisland submodule."""

import unittest

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np

import rgz.bboxes
import rgz.consensus
import rgz.constants
import rgz.units as u


def make_dummy_wcs():
    # Make a dummy WCS
    w = WCS(naxis=2)
    w.wcs.cdelt = [0.3, 0.3]
    w.wcs.crpix = [10, 15]
    w.wcs.crval = [2, 3]
    return w


class TestFindPointsInBox(unittest.TestCase):
    """Tests for rgz.bboxes.BBox.get_points_in_box."""

    def test_simple(self):
        lower_ra = lower_dec = 0.0 * u.deg
        upper_ra = upper_dec = 1.0 * u.deg
        bbox = rgz.bboxes.BBox(
            ra_min=lower_ra, ra_max=upper_ra, dec_min=lower_dec, dec_max=upper_dec
        )
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
        got = bbox.get_points_in_box(points)
        self.assertSetEqual(set(got), set(want))

    def test_ra_boundary(self):
        lower_dec = -1.0 * u.deg
        upper_dec = 1.0 * u.deg
        lower_ra = 359.9 * u.deg
        upper_ra = 0.1 * u.deg
        bbox = rgz.bboxes.BBox(
            ra_min=lower_ra, ra_max=upper_ra, dec_min=lower_dec, dec_max=upper_dec
        )
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
        got = bbox.get_points_in_box(points)
        self.assertSetEqual(set(got), set(want))

    def test_all_pix2world(self):
        """Sanity check for WCS.all_pix2world."""
        # Create some dummy RA/Dec pairs and check that the common-sense
        # transformation gives the same result as w.all_pix2world
        coords_phys = np.array(
            [
                [1, 5],
                [-1, 10],
                [99, 100],
                [256, 0.3],
                [-99, 0.0003],
                [359.0, 179.0],
                [1.3454, 63.0324],
            ]
        )
        # NOTE: w.wcs.crpix is indexed from 1, so we have to subtract 1 as
        # below.
        w = make_dummy_wcs()
        coords_px = (coords_phys - w.wcs.crval) / w.wcs.cdelt + (w.wcs.crpix - 1)
        want = coords_phys
        got = w.all_pix2world(coords_px, 0)
        self.assertTrue(np.allclose(want, got))


class TestBBox(unittest.TestCase):
    """Tests for the BBox class."""

    def test_init(self):
        # Test invalid inputs
        with self.assertRaises(ValueError):
            for args in (
                [10, 14, 50, 55],
                [10 * u.deg, 14 * u.deg, 50 * u.deg, 55 * u.deg],
                [14 * u.deg, 10 * u.deg, 50 * u.deg, 55 * u.deg],
                [10 * u.deg, 14 * u.deg, 50 * u.deg, 49 * u.deg],
                [10 * u.deg, 10 * u.deg, 50 * u.deg, 55 * u.deg],
                [10 * u.deg, 14 * u.deg, 50 * u.deg, 50 * u.deg],
                [10 * u.deg, 14 * u.deg, 87.9 * u.deg, 90.1 * u.deg],
                [10 * u.deg, 14 * u.deg, -101.9 * u.deg, -89.0 * u.deg],
                [-0.1 * u.deg, 1.2 * u.deg, 50 * u.deg, 55 * u.deg],
                [359.8 * u.deg, 361.2 * u.deg, 50 * u.deg, 55 * u.deg],
            ):
                rgz.bboxes.BBox(*args)

        # Test width, height and centre attributes are correctly populated
        bbox = rgz.bboxes.BBox(
            ra_min=59.8 * u.deg,
            ra_max=62.1 * u.deg,
            dec_min=-3.5 * u.deg,
            dec_max=-1.9 * u.deg,
        )
        self.assertTrue(bbox.width.value, (62.1 - 59.8))
        self.assertTrue(bbox.height.value, (-3.5 - -1.9))
        self.assertTrue(
            bbox.centre, SkyCoord((59.8 + 62.1) / 2 * u.deg, (-3.5 + -1.9) / 2 * u.deg)
        )

    def test_serialisation(self):
        """Test BBox.to_dict()."""
        ra_min = 10
        dec_min = 5
        ra_max = 12
        dec_max = 8
        bbox = rgz.bboxes.BBox(
            ra_min=ra_min * u.deg,
            ra_max=ra_max * u.deg,
            dec_min=dec_min * u.deg,
            dec_max=dec_max * u.deg,
        )
        bbox_dict = dict(
            ra_min=ra_min,
            ra_max=ra_max,
            dec_min=dec_min,
            dec_max=dec_max,
            width=ra_max - ra_min,
            height=dec_max - dec_min,
            centre=[
                0.5 * (ra_min + ra_max),
                0.5 * (dec_min + dec_max),
            ],
        )

        # Test serialisation
        want = bbox_dict
        got = bbox.to_json()
        self.assertDictEqual(want, got)

        # Test deserialisation
        want = bbox
        got = rgz.bboxes.BBox.from_json(bbox_dict)
        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
