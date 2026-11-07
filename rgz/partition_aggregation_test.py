"""Tests for pure partition aggregation."""

import unittest

from rgz.classifications import Classification
from rgz import partition_aggregation


class TestPartitionAggregation(unittest.TestCase):
    """Tests for rgz.partition_aggregation."""

    def test_aggregate_subset(self):
        """Tests mathematical partition aggregation on classifications."""
        classifications = [
            Classification(
                cid="1",
                zid="z1",
                username="u1",
                notes=[],
                coord_matches=[("ir1", ["A", "B"]), ("ir2", ["C"])],
            ),
            Classification(
                cid="2",
                zid="z1",
                username="u2",
                notes=[],
                coord_matches=[("ir1", ["A", "B"]), ("ir2", ["C"])],
            ),
            Classification(
                cid="3",
                zid="z1",
                username="u3",
                notes=[],
                coord_matches=[("ir1", ["A"]), ("ir2", ["B", "C"])],
            ),
        ]

        aggregation = partition_aggregation.aggregate_subset(classifications)
        expected_blocks = frozenset({frozenset({"A", "B"}), frozenset({"C"})})
        self.assertEqual(aggregation.aggregation, expected_blocks)
        self.assertIn(frozenset({"A", "B"}), aggregation.scores)
        self.assertIn(frozenset({"C"}), aggregation.scores)


if __name__ == "__main__":
    unittest.main()
