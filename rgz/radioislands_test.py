"""Tests for processing the RGZ radioisland submodule."""

import unittest

from astropy.wcs import WCS
import numpy as np

import rgz.consensus
import rgz.constants
import rgz.radioislands
import rgz.units as u


def make_dummy_wcs():
    # Make a dummy WCS
    w = WCS(naxis=2)
    w.wcs.cdelt = [0.3, 0.3]
    w.wcs.crpix = [10, 15]
    w.wcs.crval = [2, 3]
    return w


class TestFindPointsInBox(unittest.TestCase):
    """Tests for rgz.radioislands.find_points_in_box."""

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
        got = rgz.radioislands.find_points_in_box(
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
        got = rgz.radioislands.find_points_in_box(
            points, lower_ra, upper_ra, lower_dec, upper_dec
        )
        self.assertSetEqual(set(got), set(want))


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

    def test_transform_coord_radio(self):
        w = make_dummy_wcs()

        # Check for coords being out of bounds
        with self.assertRaises(ValueError):
            rgz.radioislands.transform_coord_radio(np.array([-1, 0]), w)
            rgz.radioislands.transform_coord_radio(np.array([132, 0]), w)
            rgz.radioislands.transform_coord_radio(np.array([-1, 253]), w)

        # Check transformation has worked correctly
        for coords in [[0, 0], [1, 10], [99, 100], [131, 5]]:
            coords_unscaled_px = np.array([coords])
            coords_px = coords_unscaled_px / rgz.constants.RADIO_MAX_PX * 100
            coords_phys = (coords_px - (w.wcs.crpix - 1)) * w.wcs.cdelt + w.wcs.crval
            want = coords_phys
            got = rgz.radioislands.transform_coord_radio(coords_unscaled_px[0], wcs=w)
            self.assertTrue(np.allclose(want, np.array([[c.value for c in got]])))


class TestTransformBboxPxToPhys(unittest.TestCase):
    """Tests for transform_bbox_px_to_phys."""

    # TODO(hzovaro): update once bboxes are refactored

    def test_transform(self):
        # TODO(hzovaro) test some more coords
        xmin = 1
        ymin = 1
        xmax = 2
        ymax = 2
        bbox = [xmin, ymin, xmax, ymax]
        w = make_dummy_wcs()
        want = np.concatenate(
            [
                rgz.radioislands.transform_coord_radio(
                    coord=np.array([xmin, 131 - ymax]), wcs=w
                ),
                rgz.radioislands.transform_coord_radio(
                    coord=np.array([xmax, 131 - ymin]), wcs=w
                ),
            ]
        )
        got = rgz.radioislands.transform_bbox_px_to_phys(px_bbox=bbox, wcs=w)
        self.assertTrue(np.allclose(want, got))


if __name__ == "__main__":
    unittest.main()
