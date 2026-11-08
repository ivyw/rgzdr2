"""Tests for aggregating RGZ classifications."""

import datetime
import json
import logging
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from rgz import consensus
from rgz import testutils

logger = logging.getLogger(__name__)

_MAX_RUNTIME_PER_DR1_CLASSIFICATION = datetime.timedelta(microseconds=50)


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

    def test_runtime(self):
        """Performance test for DR1 consensus aggregation."""
        n_copies = 20
        with open(
            self.test_data_path / testutils.SUBJECTS_PROCESSED_FILENAME,
            encoding="utf-8",
        ) as f:
            base_subjects = json.load(f)
        with open(
            self.test_data_path / testutils.CLASSIFICATIONS_MATCHED_FILENAME,
            encoding="utf-8",
        ) as f:
            base_classifications = json.load(f)

        subjects_data = [
            {**s, "zid": f"{s['zid']}_{i}"}
            for i in range(n_copies)
            for s in base_subjects
        ]
        classifications_data = [
            {**c, "zid": f"{c['zid']}_{i}"}
            for i in range(n_copies)
            for c in base_classifications
        ]

        subjects_path = self.temp_dir_path / "subjects_20x.json"
        classifications_path = self.temp_dir_path / "classifications_20x.json"
        output_path = self.temp_dir_path / "consensus_runtime_out.json"
        with open(subjects_path, "w", encoding="utf-8") as f:
            json.dump(subjects_data, f)
        with open(classifications_path, "w", encoding="utf-8") as f:
            json.dump(classifications_data, f)

        now = time.perf_counter()
        consensus.aggregate(subjects_path, classifications_path, output_path)
        elapsed = datetime.timedelta(seconds=time.perf_counter() - now)
        elapsed_per_classification = elapsed / len(classifications_data)
        logger.info(
            "consensus.aggregate (DR1): %.2f us/classification (%d classifications in %.3fs)",
            elapsed_per_classification / datetime.timedelta(microseconds=1),
            len(classifications_data),
            elapsed.total_seconds(),
        )

        self.assertLess(
            elapsed_per_classification,
            _MAX_RUNTIME_PER_DR1_CLASSIFICATION,
            f"consensus.aggregate took {elapsed_per_classification} per classification "
            f"(> {_MAX_RUNTIME_PER_DR1_CLASSIFICATION})",
        )


if __name__ == "__main__":
    unittest.main()
