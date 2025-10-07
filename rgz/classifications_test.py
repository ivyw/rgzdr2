"""Tests for processing RGZ classifications."""

import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest

import rgz.classifications

# "Cache" data filename.
_TEST_CACHE_DATA_FILENAME = "first"

# Test (processed) subjects JSON filename.
_TEST_SUBJECTS_PROCESSED_FILENAME = "radio_subjects_test_subset_processed.json"

# Test (raw) classifications JSON filename.
_TEST_CLASSIFICATIONS_FILENAME = "radio_classifications_test_subset.json"

# Test (processed) classifications JSON filename.
_TEST_CLASSIFICATIONS_PROCESSED_FILENAME = (
    "radio_classifications_test_subset_processed.json"
)


def get_test_data_dir() -> Path:
    """Gets the directory that test data is held in."""
    current_frame = inspect.currentframe()
    if current_frame is not None:
        current_file_path = inspect.getfile(current_frame)
    else:
        current_file_path = __file__
    return Path(os.path.dirname(current_file_path)) / "testdata"


class TestProcess(unittest.TestCase):
    """Tests for rgz.classifications.process."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.test_data_path = get_test_data_dir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self):
        """Tests behaviour consistency in processing classifications."""
        output_path = self.temp_dir_path / "out.json"
        rgz.classifications.process(
            self.test_data_path / _TEST_CLASSIFICATIONS_FILENAME,
            self.test_data_path / _TEST_SUBJECTS_PROCESSED_FILENAME,
            self.test_data_path / _TEST_CACHE_DATA_FILENAME,
            output_path,
        )
        with open(output_path) as f:
            got = json.load(f)
        with open(self.test_data_path / _TEST_CLASSIFICATIONS_PROCESSED_FILENAME) as f:
            want = json.load(f)
        self.assertEqual(want, got)


class TestClassification(unittest.TestCase):
    """Tests for rgz.classification.Classification."""

    def test_radio_combinations(self):
        cl = rgz.classifications.Classification(
            cid="123",
            zid="ARG123",
            coord_matches=[
                (
                    "01 01 01 +01 01 01",
                    rgz.classifications.RadioSource(["FIRST1234", "FIRST3456"]),
                ),
                (
                    "02 01 01 +01 01 01",
                    rgz.classifications.RadioSource(["FIRST56"]),
                ),
            ],
            username="",
            notes=[],
        )
        got = cl.radio_combinations()
        want = rgz.classifications.RadioSourceCombination(
            (("FIRST1234", "FIRST3456"), ("FIRST56",))
        )
        self.assertEqual(want, got)


class TestRadioCombination(unittest.TestCase):
    """Tests for rgz.consensus.RadioCombination."""

    def test_invariant(self):
        """Same radio sources produce same RadioCombination."""
        ordered_radio_sources = [
            [["abc"], ["def", "ghi"], ["jkl", "hij", "hij"]],
            [["abc"], ["ghi", "def"], ["hij", "jkl", "hij"]],
            [["hij", "jkl", "hij"], ["abc"], ["ghi", "def"]],
            [["hij", "jkl"], ["abc"], ["ghi", "def"]],
        ]

        combinations = []
        for radio_source in ordered_radio_sources:
            combinations.append(
                rgz.classifications.RadioSourceCombination(radio_source)
            )

        comparison_combination = combinations[0]
        for radio_combination in combinations[1:]:
            self.assertEqual(comparison_combination, radio_combination)

    def test_sources(self):
        """Returns correct sources."""
        input_sources = [["abc"], ["def", "ghi"], ["jkl", "hij", "hij"]]
        radio_combination = rgz.classifications.RadioSourceCombination(input_sources)
        got = radio_combination.sources()
        want = frozenset(
            {
                rgz.classifications.RadioSource({"abc"}),
                rgz.classifications.RadioSource({"def", "ghi"}),
                rgz.classifications.RadioSource({"jkl", "hij"}),
            }
        )
        self.assertEqual(got, want)


class TestRadioSource(unittest.TestCase):
    """Tests for rgz.consensus.RadioSource."""

    def test_invariant(self):
        """Same radio sources produce same RadioSource."""
        ordered_radio_sources = [
            ["abc"],
            ["def", "ghi"],
            ["jkl", "hij", "hij"],
            ["ghi", "def"],
            ["hij", "jkl", "hij"],
            [],
        ]
        want = [
            frozenset({"abc"}),
            frozenset({"def", "ghi"}),
            frozenset({"jkl", "hij"}),
            frozenset({"def", "ghi"}),
            frozenset({"jkl", "hij"}),
            frozenset(),
        ]

        for input, want_ in zip(ordered_radio_sources, want):
            self.assertEqual(
                want_,
                rgz.classifications.RadioSource(input).components(),
            )

    def test_sources(self):
        """Returns correct sources."""
        input_sources = [["abc"], ["def", "ghi"], ["jkl", "hij", "hij"]]
        radio_combination = rgz.classifications.RadioSourceCombination(input_sources)
        got = radio_combination.sources()
        want = frozenset(
            {
                rgz.classifications.RadioSource({"abc"}),
                rgz.classifications.RadioSource({"def", "ghi"}),
                rgz.classifications.RadioSource({"jkl", "hij"}),
            }
        )
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
