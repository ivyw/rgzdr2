"""Tests for processing the RGZ radioisland submodule."""

import unittest

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np

import rgz.consensus
import rgz.constants
import rgz.radio_islands
import rgz.units as u


def make_dummy_wcs():
    # Make a dummy WCS
    w = WCS(naxis=2)
    w.wcs.cdelt = [0.3, 0.3]
    w.wcs.crpix = [10, 15]
    w.wcs.crval = [2, 3]
    return w


class TestFindPointsInBox(unittest.TestCase):
    """Tests for rgz.radio_islands.find_points_in_box."""

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
        got = rgz.radio_islands.find_points_in_box(
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
        got = rgz.radio_islands.find_points_in_box(
            points, lower_ra, upper_ra, lower_dec, upper_dec
        )
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


class TestTransformBboxPxToPhys(unittest.TestCase):
    """Tests for transform_rgzbbox_to_phys."""

    @unittest.skip("transform_coord_radio has been removed")
    def test_transform(self):
        # TODO(hzovaro) Replace transform_coord_radio 
        xmin = 1
        ymin = 2
        xmax = 2
        ymax = 1
        bbox = (xmin, ymin, xmax, ymax)
        w = make_dummy_wcs()
        want = np.concatenate(
            [
                rgz.radio_islands.transform_coord_radio(
                    coord=np.array([xmin - 1, 131 - (ymin - 1)]),
                    wcs=w,  # = 131 - 1 = 130
                ),
                rgz.radio_islands.transform_coord_radio(
                    coord=np.array([xmax - 1, 131 - (ymax - 1)]),
                    wcs=w,  # = 131 - 0 = 131
                ),
            ]
        )
        want = rgz.radio_islands.BBox(
            xmin=want[0],
            ymin=want[1],
            xmax=want[2],
            ymax=want[3],
        )

        got = rgz.radio_islands.transform_rgzbbox_to_phys(bbox=bbox, wcs=w)
        self.assertEqual(want, got)


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
                rgz.radio_islands.BBox(*args)

        # Test width, height and centre attributes are correctly populated
        bbox = rgz.radio_islands.BBox(
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
        bbox = rgz.radio_islands.BBox(
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
        got = rgz.radio_islands.BBox.from_json(bbox_dict)
        self.assertEqual(want, got)

class TestRadioIsland(unittest.TestCase):

    def test_inputs(self):
        """Tests for input validation."""
        # TODO(hzovaro): implement these

    def test_eq(self):
        """Test the __eq__ method."""
        rng = np.random.RandomState(42)

        ra_min_1 = 10
        dec_min_1 = 5
        ra_max_1 = 12
        dec_max_1 = 8
        bbox_1 = rgz.radio_islands.BBox(
            ra_min=ra_min_1 * u.deg,
            ra_max=ra_max_1 * u.deg,
            dec_min=dec_min_1 * u.deg,
            dec_max=dec_max_1 * u.deg,
        )
        rgzbbox_1 = [1, 2, 3, 4]
        firsts_1 = ["FIRST1", "FIRST2", "FIRST3"]
        contours_1 = [
            [(ra, dec) for ra, dec in zip(
                rng.uniform(low=50, high=51, size=25),
                rng.uniform(low=-45, high=-44, size=25))],
            [(ra, dec) for ra, dec in zip(
                rng.uniform(low=50, high=51, size=25),
                rng.uniform(low=-45, high=-44, size=25))],
        ]

        ri_1 = rgz.radio_islands.RadioIsland(
            bbox=bbox_1, rgzbbox=rgzbbox_1, contours=contours_1, firsts=firsts_1
        )
        
        ra_min_2 = 25
        dec_min_2 = 16
        ra_max_2 = 33.5
        dec_max_2 = 25
        bbox_2 = rgz.radio_islands.BBox(
            ra_min=ra_min_2 * u.deg,
            ra_max=ra_max_2 * u.deg,
            dec_min=dec_min_2 * u.deg,
            dec_max=dec_max_2 * u.deg,
        )
        rgzbbox_2 = [1, 2, 3, 4]
        firsts_2 = ["FIRST1", "FIRST2", None]
        contours_2 = [
            [(0.1, 1.3), (0.2, 1.3), (0.05, 0.9), (0.5, 1.5), (0.1, np.nan),],
            [(0.1, np.nan), (0.2, 1.3), (0.05, 0.9), (0.5, 1.5), (0.1, np.inf),],
        ]

        ri_2 = rgz.radio_islands.RadioIsland(
            bbox=bbox_2, rgzbbox=rgzbbox_2, contours=contours_2, firsts=firsts_2
        )

        self.assertTrue(ri_1 == ri_1)
        self.assertTrue(ri_2 == ri_2)
        self.assertFalse(ri_1 == ri_2)


    def test_serialisation(self):
        """Tests for the to_json() and from_json() methods."""
        rng = np.random.RandomState(42)
        ra_min = 10
        dec_min = 5
        ra_max = 12
        dec_max = 8
        bbox = rgz.radio_islands.BBox(
            ra_min=ra_min * u.deg,
            ra_max=ra_max * u.deg,
            dec_min=dec_min * u.deg,
            dec_max=dec_max * u.deg,
        )
        rgzbbox = [1, 2, 3, 4]
        firsts = ["FIRST1", "FIRST2", "FIRST3"]
        contours = [
            [(ra, dec) for ra, dec in zip(
                rng.uniform(low=50, high=51, size=25),
                rng.uniform(low=-45, high=-44, size=25))],
            [(ra, dec) for ra, dec in zip(
                rng.uniform(low=50, high=51, size=25),
                rng.uniform(low=-45, high=-44, size=25))],
        ]

        ri = rgz.radio_islands.RadioIsland(
            bbox=bbox, rgzbbox=rgzbbox, contours=contours, firsts=firsts
        )
        ri_dict = dict(
            bbox=bbox.to_json(),
            rgzbbox=rgzbbox,
            contours=contours,
            firsts=firsts,
        )

        # Serialisation
        want = ri_dict 
        got = ri.to_json()
        self.assertEqual(want, got)
        
        # Deserialisation
        want = ri 
        got = rgz.radio_islands.RadioIsland.from_json(ri_dict)
        self.assertEqual(want.bbox, got.bbox)
        self.assertEqual(want.rgzbbox, got.rgzbbox)
        self.assertEqual(want.firsts, got.firsts)
        self.assertEqual(want.contours, got.contours)


if __name__ == "__main__":
    unittest.main()
