"""Tests for processing RGZ classifications."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import astropy.table
from astropy.coordinates import SkyCoord

import rgz.classifications
import rgz.testutils

class TestProcess(unittest.TestCase):
    """Tests for rgz.classifications.process."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.test_data_path = rgz.testutils.get_test_data_dir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_regression(self, update: bool = False):
        """Tests behaviour consistency in processing classifications."""
        output_path = self.temp_dir_path / "out.json"
        rgz.classifications.process(
            self.test_data_path / rgz.testutils.CLASSIFICATIONS_FILENAME,
            self.test_data_path / rgz.testutils.SUBJECTS_PROCESSED_FILENAME,
            output_path,
        )
        with open(output_path) as f:
            got = json.load(f)

        want_path = (
            self.test_data_path / rgz.testutils.CLASSIFICATIONS_PROCESSED_FILENAME
        )

        if update:
            with open(want_path, "w") as f:
                json.dump(got, f)

        with open(want_path) as f:
            want = json.load(f)
        print(got[0])
        print(want[0])
        for gg, ww in zip(got, want):
            # drop NOFIRSTS as these will never be the same due to rounding changes
            for cm in ww["coord_matches"]:
                cm["radio"] = [
                    "NOFIRST" for r in cm["radio"] if r.startswith("NOFIRST")
                ]
            for cm in gg["coord_matches"]:
                cm["radio"] = [
                    "NOFIRST" for r in cm["radio"] if r.startswith("NOFIRST")
                ]
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


class TestHostLookup(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        self.test_data_path = rgz.testutils.get_test_data_dir()

        self.patcher_pool = mock.patch("rgz.classifications.multiprocessing.Pool")
        self.patcher_query_irsa = mock.patch("rgz.classifications.query_irsa")

        self.mock_pool = self.patcher_pool.start()
        self.mock_query_irsa = self.patcher_query_irsa.start()

        def fake_query_irsa(radius, coordinates_to_lookup):
            return rgz.testutils.get_wise_irsa_query_result()

        self.mock_query_irsa.side_effect = fake_query_irsa

        class FakePool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def starmap(self, func, iterable):
                return [func(*args) for args in iterable]

        self.mock_pool.return_value = FakePool()

    def tearDown(self):
        self.temp_dir.cleanup()
        self.patcher_pool.stop()
        self.patcher_query_irsa.stop()

    def test_host_lookup(self):
        """Checks host_lookup adds ir_matches without altering existing fields."""
        input_path = (
            self.test_data_path / rgz.testutils.CLASSIFICATIONS_PROCESSED_FILENAME
        )
        output_path = self.temp_dir_path / "host_lookup_out.json"

        rgz.classifications.host_lookup(input_path, output_path)

        with open(input_path) as f:
            want = json.load(f)
        with open(output_path) as f:
            got = json.load(f)

        self.assertEqual(len(got), len(want))
        for got_classification, want_classification in zip(got, want):
            self.assertIn("ir_matches", got_classification)
            self.assertEqual(
                len(got_classification["ir_matches"]),
                len(want_classification["coord_matches"]),
            )
            del got_classification["ir_matches"]
            self.assertEqual(got_classification, want_classification)

    def test_regression(self, update: bool = False):
        """Tests behaviour consistency in matching classifications."""
        output_path = self.temp_dir_path / "out.json"
        rgz.classifications.host_lookup(
            self.test_data_path / rgz.testutils.CLASSIFICATIONS_PROCESSED_FILENAME,
            output_path,
        )
        with open(output_path) as f:
            got = json.load(f)

        want_path = (
            self.test_data_path / rgz.testutils.CLASSIFICATIONS_MATCHED_FILENAME
        )

        if update:
            with open(want_path, "w") as f:
                json.dump(got, f)

        with open(want_path) as f:
            want = json.load(f)
        
        self.assertEqual(len(got), len(want))
        for got_, want_ in zip(got, want):
            # Improve error messages by checking each entry individually.
            rgz.testutils.assert_json_almost_equal(self, want_, got_)


if __name__ == "__main__":
    unittest.main()
