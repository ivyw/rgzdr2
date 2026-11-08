"""Tests for pure partition aggregation."""

import datetime
import logging
import math
import time
import unittest

from rgz.classifications import Classification
from rgz import partition_aggregation

logger = logging.getLogger(__name__)

_MAX_RUNTIME_PER_AGGREGATE_SUBSET_CALL = datetime.timedelta(milliseconds=4)


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

    def test_large_unanimous_clique_no_overflow(self):
        """Tests that a 10-element unanimous clique does not overflow float64."""
        items = [f"A{i}" for i in range(10)]
        classifications = [
            Classification(
                cid=str(i),
                zid="z1",
                username=f"u{i}",
                notes=[],
                coord_matches=[("ir1", items)],
            )
            for i in range(5)
        ]

        aggregation = partition_aggregation.aggregate_subset(classifications)
        expected_block = frozenset(items)
        self.assertEqual(aggregation.aggregation, frozenset({expected_block}))
        self.assertFalse(math.isnan(aggregation.probability))
        self.assertAlmostEqual(aggregation.probability, 1.0)
        self.assertAlmostEqual(aggregation.scores[expected_block], 1.0)

    def test_runtime(self):
        """Performance test for aggregate_subset on an 8-element clique."""
        classifications = [
            Classification(
                cid=str(i),
                zid="z1",
                username=f"u{i}",
                notes=[],
                coord_matches=(
                    [
                        ("ir1", ["A", "B", "C"]),
                        ("ir2", ["D", "E"]),
                        ("ir3", ["F", "G", "H"]),
                    ]
                    if i % 3 != 0
                    else [
                        ("ir1", ["A", "B"]),
                        ("ir2", ["C", "D", "E"]),
                        ("ir3", ["F", "G"]),
                        ("ir4", ["H"]),
                    ]
                ),
            )
            for i in range(20)
        ]

        n_calls = 20
        now = time.perf_counter()
        for _ in range(n_calls):
            partition_aggregation.aggregate_subset(classifications)
        elapsed = datetime.timedelta(seconds=time.perf_counter() - now)
        elapsed_per_call = elapsed / n_calls
        logger.info(
            "aggregate_subset: %.2f ms/call (%d calls in %.3fs)",
            elapsed_per_call / datetime.timedelta(milliseconds=1),
            n_calls,
            elapsed.total_seconds(),
        )

        self.assertLess(
            elapsed_per_call,
            _MAX_RUNTIME_PER_AGGREGATE_SUBSET_CALL,
            f"aggregate_subset took {elapsed_per_call} per call "
            f"(> {_MAX_RUNTIME_PER_AGGREGATE_SUBSET_CALL})",
        )


if __name__ == "__main__":
    unittest.main()
