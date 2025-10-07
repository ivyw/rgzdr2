"""Tests for aggregating RGZ classifications."""

import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest

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
        want_hosts = set(c["host_name"] for c in want)
        got_hosts = set(c["host_name"] for c in got)
        self.assertEqual(want_hosts, got_hosts)
        # Check contained hosts (order-dependent).
        want_hosts = [c["host_name"] for c in want]
        got_hosts = [c["host_name"] for c in got]
        self.assertEqual(want_hosts, got_hosts)

        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
