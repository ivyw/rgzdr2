"""Tests for processing RGZ subjects."""

import datetime
import json
import logging
from pathlib import Path
import tempfile
import time
import unittest

import rgz.consensus
import rgz.constants
import rgz.first
import rgz.subjects
from rgz import testutils

logger = logging.getLogger(__name__)


# Path to test directory.
_TEST_DIR = testutils.get_test_data_dir()

# Path to "cache" data.
_TEST_CACHE_DATA_PATH = _TEST_DIR / testutils.CACHE_DATA_FILENAME

# Path to test (raw) subjects JSON.
_TEST_SUBJECTS_PATH = _TEST_DIR / "subjects.json"

# Path to test (processed) subjects JSON.
_TEST_SUBJECTS_PROCESSED_PATH = _TEST_DIR / testutils.SUBJECTS_PROCESSED_FILENAME

_MAX_RUNTIME_PER_PROCESS_CALL = datetime.timedelta(milliseconds=6.5)


class TestProcess(unittest.TestCase):
    """Tests for rgz.subjects.process."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self, update: bool = False):
        """Tests behaviour consistency in processing subjects."""
        output_path = self.temp_dir_path / "out.json"
        rgz.subjects.process(_TEST_SUBJECTS_PATH, _TEST_CACHE_DATA_PATH, output_path)
        with open(output_path) as f:
            got = json.load(f)

        if update:
            with open(_TEST_SUBJECTS_PROCESSED_PATH, "w") as f:
                json.dump(got, f)

        with open(_TEST_SUBJECTS_PROCESSED_PATH) as f:
            want = json.load(f)

        self.assertEqual(len(got), len(want))
        for got_, want_ in zip(got, want):
            # Improve error messages by checking each entry individually.
            testutils.assert_json_almost_equal(self, want_, got_)

    def test_runtime(self):
        """Performance test for processing subjects."""
        first_catalogue = rgz.first.fetch_first_catalogue_from_server_or_cache(
            _TEST_CACHE_DATA_PATH
        )
        first_tree = rgz.first.build_first_tree(first_catalogue)

        with open(_TEST_SUBJECTS_PATH, encoding="utf-8") as f:
            raw_subjects = [json.loads(row) for row in f] * 10

        now = time.perf_counter()
        for raw_subject in raw_subjects:
            rgz.subjects.process_subject(
                raw_subject, _TEST_CACHE_DATA_PATH, first_tree
            )
        elapsed = datetime.timedelta(seconds=time.perf_counter() - now)
        elapsed_per_call = elapsed / len(raw_subjects)
        logger.info(
            "process_subject: %.2f ms/call (%d calls in %.3fs)",
            elapsed_per_call / datetime.timedelta(milliseconds=1),
            len(raw_subjects),
            elapsed.total_seconds(),
        )

        self.assertLess(
            elapsed_per_call,
            _MAX_RUNTIME_PER_PROCESS_CALL,
            f"Processing subjects took {elapsed_per_call} per call (> {_MAX_RUNTIME_PER_PROCESS_CALL})",
        )


if __name__ == "__main__":
    unittest.main()
