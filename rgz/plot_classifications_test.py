import json
import logging
from pathlib import Path
import unittest

import matplotlib.pyplot as plt

from rgz import classifications
from rgz import plot_classifications
from rgz import subjects


logging = logging.getLogger(__name__)

@unittest.skip("not implemented")
class TestPlotClassifications(unittest.TestCase):

    def test_plot_classifications(self):
        raise NotImplementedError


    def test_subject_classification_mismatch(self):
        raise NotImplementedError


    def test_pixel_scaling(self):
        # TODO(hzovaro) make a tmp JSON file containing coordinates. Apply a custom 
        # stretch and check that the outputs are as expected
        raise NotImplementedError
        


if __name__ == "__main__":
    unittest.main()

    # Paths
    testdata_path = Path("testdata")
    raw_subjects_path = testdata_path / "radio_subjects_test_subset.json"
    processed_subjects_path = (
        testdata_path / "radio_subjects_test_subset_processed.json"
    )
    processed_classifications_path = (
        testdata_path / "radio_classifications_test_subset_processed.json"
    )

    # Load a subject
    subject_idx = 2
    with open(processed_subjects_path, "r") as f:
        subject_json = json.load(f)[subject_idx]
    subject = subjects.Subject.from_json(subject_json)
    logging.info(subject.id)

    # Find a classification that references this subject
    with open(processed_classifications_path, "r") as f:
        classifications_json = [c for c in json.load(f) if c["zid"] == subject.zid]
    classifications_list = [
        classifications.Classification.from_json(cs) for cs in classifications_json
    ]

    # Test general functionality
    for ii in range(len(classifications_list))[:4]:
        ax = plot_classifications.plot_single_classification(
            classification=classifications_list[ii],
            subject=subject,
            cache=testdata_path / "first",
            ax=None,
        )

    # Test that existing axes are destroyed
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.3, top=0.7, bottom=0.05)
    _ = plot_classifications.plot_single_classification(
        classification=classifications_list[0],
        subject=subject,
        cache=testdata_path / "first",
        ax=ax,
    )
