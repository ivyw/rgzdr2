"""Tests for processing the RGZ radioisland submodule."""

import unittest

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import numpy as np

import rgz.bboxes
import rgz.consensus
import rgz.constants
import rgz.radio_islands
import rgz.units as u


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
        bbox_1 = rgz.bboxes.BBox(
            ra_min=ra_min_1 * u.deg,
            ra_max=ra_max_1 * u.deg,
            dec_min=dec_min_1 * u.deg,
            dec_max=dec_max_1 * u.deg,
        )
        rgzbbox_1 = [1, 2, 3, 4]
        firsts_1 = ["FIRST1", "FIRST2", "FIRST3"]
        contours_1 = [
            [
                (ra, dec)
                for ra, dec in zip(
                    rng.uniform(low=50, high=51, size=25),
                    rng.uniform(low=-45, high=-44, size=25),
                )
            ],
            [
                (ra, dec)
                for ra, dec in zip(
                    rng.uniform(low=50, high=51, size=25),
                    rng.uniform(low=-45, high=-44, size=25),
                )
            ],
        ]

        ri_1 = rgz.radio_islands.RadioIsland(
            bbox=bbox_1, rgzbbox=rgzbbox_1, contours=contours_1, firsts=firsts_1
        )

        ra_min_2 = 25
        dec_min_2 = 16
        ra_max_2 = 33.5
        dec_max_2 = 25
        bbox_2 = rgz.bboxes.BBox(
            ra_min=ra_min_2 * u.deg,
            ra_max=ra_max_2 * u.deg,
            dec_min=dec_min_2 * u.deg,
            dec_max=dec_max_2 * u.deg,
        )
        rgzbbox_2 = [1, 2, 3, 4]
        firsts_2 = ["FIRST1", "FIRST2", None]
        contours_2 = [
            [
                (0.1, 1.3),
                (0.2, 1.3),
                (0.05, 0.9),
                (0.5, 1.5),
                (0.1, np.nan),
            ],
            [
                (0.1, np.nan),
                (0.2, 1.3),
                (0.05, 0.9),
                (0.5, 1.5),
                (0.1, np.inf),
            ],
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
        bbox = rgz.bboxes.BBox(
            ra_min=ra_min * u.deg,
            ra_max=ra_max * u.deg,
            dec_min=dec_min * u.deg,
            dec_max=dec_max * u.deg,
        )
        rgzbbox = [1, 2, 3, 4]
        firsts = ["FIRST1", "FIRST2", "FIRST3"]
        contours = [
            [
                (ra, dec)
                for ra, dec in zip(
                    rng.uniform(low=50, high=51, size=25),
                    rng.uniform(low=-45, high=-44, size=25),
                )
            ],
            [
                (ra, dec)
                for ra, dec in zip(
                    rng.uniform(low=50, high=51, size=25),
                    rng.uniform(low=-45, high=-44, size=25),
                )
            ],
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
