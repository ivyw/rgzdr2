"""Tests for processing RGZ subjects."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from python.runfiles import Runfiles
import line_profiler

import rgz.subjects

# Path to test directory.
_TEST_DIR = Path(os.path.dirname(__file__)) / "testdata/"

# Path to "cache" data.
_TEST_CACHE_DATA_PATH = _TEST_DIR / "first"

# Path to test (raw) subjects JSON.
_TEST_SUBJECTS_PATH = _TEST_DIR / "radio_subjects_test_subset.json"

# Path to test (processed) subjects JSON.
_TEST_SUBJECTS_PROCESSED_PATH = _TEST_DIR / "radio_subjects_test_subset_processed.json"


profiler = line_profiler.LineProfiler()
profiler.enable_by_count()

class TestProcess(unittest.TestCase):
    """Tests for rgz.subjects.process."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        profiler.add_module(rgz.subjects)
        profiler.add_function(cls.test_regression)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @classmethod
    def tearDownClass(cls):  
        profiler.print_stats()
        profiler.dump_stats("/tmp/lprof")

    def test_regression(self):
        """Tests behaviour consistency in processing subjects."""
        output_path = self.temp_dir_path / "out.json"
        rgz.subjects.process(_TEST_SUBJECTS_PATH, _TEST_CACHE_DATA_PATH, output_path)
        with open(output_path) as f:
            got = json.load(f)
        with open(_TEST_SUBJECTS_PROCESSED_PATH) as f:
            want = json.load(f)
        self.assertEqual(want, got)


if __name__ == "__main__":
    unittest.main()
