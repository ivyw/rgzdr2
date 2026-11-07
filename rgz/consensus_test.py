"""Tests for aggregating RGZ classifications."""

import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from rgz import consensus
from rgz import testutils


class TestAggregate(unittest.TestCase):
    """Tests for rgz.consensus.aggregate."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.test_data_path = testutils.get_test_data_dir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self):
        """Tests behaviour consistency in aggregating classifications."""
        output_path = self.temp_dir_path / "out.json"
        consensus.aggregate(
            self.test_data_path / testutils.SUBJECTS_PROCESSED_FILENAME,
            self.test_data_path / testutils.CLASSIFICATIONS_MATCHED_FILENAME,
            output_path,
        )
        with open(output_path) as f:
            got = json.load(f)
        with open(self.test_data_path / testutils.CONSENSUS_FILENAME) as f:
            want = json.load(f)

        self.maxDiff = None

        # We have a few different comparisons here to get useful test outputs for failure.

        # Check order of subjects.
        want_zids = [c["zid"] for c in want]
        got_zids = [c["zid"] for c in got]
        self.assertEqual(want_zids, got_zids)
        # Check contained hosts (order-independent).
        want_hosts = {c["host_name"] for c in want}
        got_hosts = {c["host_name"] for c in got}
        self.assertEqual(want_hosts, got_hosts)
        # Check contained hosts (order-dependent).
        want_hosts = [c["host_name"] for c in want]
        got_hosts = [c["host_name"] for c in got]
        self.assertEqual(want_hosts, got_hosts)

        for want_, got_ in zip(want, got):
            self.assertEqual(want_, got_)

    def test_regression_partition(self):
        """Tests behaviour consistency in partition-aggregating classifications."""
        output_path = self.temp_dir_path / "out.json"
        consensus.aggregate(
            self.test_data_path / testutils.SUBJECTS_PROCESSED_FILENAME,
            self.test_data_path / testutils.CLASSIFICATIONS_MATCHED_FILENAME,
            output_path,
            method=consensus.AggregationMethod.PARTITION,
        )
        with open(output_path) as f:
            got = json.load(f)
        with open(self.test_data_path / testutils.CONSENSUS_PARTITION_FILENAME) as f:
            want = json.load(f)

        self.maxDiff = None

        # Check order of subject sets.
        want_zids = [c["zids"] for c in want]
        got_zids = [c["zids"] for c in got]
        self.assertEqual(want_zids, got_zids)
        # Check order of component sets.
        want_components = [c["components"] for c in want]
        got_components = [c["components"] for c in got]
        self.assertEqual(want_components, got_components)
        # Check contained hosts (order-independent).
        want_hosts = {c["host_name"] for c in want}
        got_hosts = {c["host_name"] for c in got}
        self.assertEqual(want_hosts, got_hosts)
        # Check contained hosts (order-dependent).
        want_hosts = [c["host_name"] for c in want]
        got_hosts = [c["host_name"] for c in got]
        self.assertEqual(want_hosts, got_hosts)

        for want_, got_ in zip(want, got):
            self.assertEqual(want_.keys(), got_.keys())
            for key in want_:
                want_val, got_val = want_[key], got_[key]
                if isinstance(want_val, float) and math.isnan(want_val):
                    self.assertTrue(math.isnan(got_val))
                elif isinstance(want_val, float):
                    self.assertAlmostEqual(want_val, got_val)
                else:
                    self.assertEqual(want_val, got_val)

    def test_partition_max_subset_size_exceeded(self):
        """Tests that partition consensus errors when a clique is too big."""
        output_path = self.temp_dir_path / "out.json"
        with self.assertRaises(ValueError):
            consensus.aggregate(
                self.test_data_path / testutils.SUBJECTS_PROCESSED_FILENAME,
                self.test_data_path / testutils.CLASSIFICATIONS_MATCHED_FILENAME,
                output_path,
                method=consensus.AggregationMethod.PARTITION,
                max_subset_size=1,
            )

    def test_map_components_to_islands(self):
        """Tests that components in the same radio island are mapped to the same representatives."""
        subject1 = MagicMock(
            radio_islands=[
                MagicMock(firsts=["A", "B"]),
                MagicMock(firsts=["C"]),
            ]
        )
        subject2 = MagicMock(
            radio_islands=[
                MagicMock(firsts=["B", "D"]),
            ]
        )
        component_to_island = consensus.map_components_to_islands([subject1, subject2])
        self.assertEqual(component_to_island["A"], component_to_island["B"])
        self.assertEqual(component_to_island["A"], component_to_island["D"])
        self.assertNotEqual(component_to_island["A"], component_to_island["C"])


if __name__ == "__main__":
    unittest.main()
