"""Other aggregation strategies for classifications."""

from collections.abc import Collection, Iterable, Sequence
import dataclasses
import functools
import itertools

# import ClusterEnsembles as cluster_ensembles
import disjoint_set
import numpy as np

from rgz.classifications import Classification
from rgz.first import FIRSTID

# A big number for situations where everyone agrees.
VERY_BIG = 1e10


def get_all_first_ids(classifications: Sequence[Classification]) -> Collection[FIRSTID]:
    """Gets all FIRST IDs touched by a set of classifications."""
    all_names = set()
    for c in classifications:
        for _, radios in c.coord_matches:
            for r in radios:
                all_names.add(r)
    return all_names


def powerset[T](iterable: Iterable[T]) -> frozenset[frozenset[T]]:
    """Iterates over the powerset."""
    s = list(iterable)
    return frozenset(
        {
            frozenset(s)
            for s in itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(len(s) + 1)
            )
        }
    )


@dataclasses.dataclass
class Aggregation:
    aggregation: frozenset[frozenset[FIRSTID]]
    scores: dict[frozenset[FIRSTID], float]
    probability: float


def aggregate_subset_dp(
    classifications: Sequence[Classification],
) -> Aggregation:
    """Aggregates a subset of classifications using dynamic programming.

    Args:
        classifications: Subset of classifications.

    Returns:
        Consensus.
    """
    # Convert FIRST IDs into indices.
    all_names = sorted(get_all_first_ids(classifications))
    name_to_index = {name: i for i, name in enumerate(all_names)}

    n = len(all_names)
    n_observed_same = np.zeros((n, n))
    n_observed_different = np.zeros((n, n))
    for c in classifications:
        observed = set()
        ds = disjoint_set.DisjointSet()
        for _, radios in c.coord_matches:
            for r in radios:
                observed.add(r)
                for s in radios:
                    if r != s:
                        ds.union(r, s)

        # TODO: Don't double-count (the observation matrix is symmetric).
        for a in observed:
            for b in observed:
                if a == b:
                    continue

                if ds.connected(a, b):
                    n_observed_same[name_to_index[a], name_to_index[b]] += 1
                else:
                    n_observed_different[name_to_index[a], name_to_index[b]] += 1

    n_observed_total = n_observed_different + n_observed_same
    # TODO: Figure out how to handle unobserved pairs.
    # It'd be nice to use something other than a zero prior.
    # We could do a prior based on distance, perhaps.
    # This situation will fail for large galaxies where a-b and b-c are
    # seen together, but never a-c. This effectively hard-caps our
    # galaxies at 4.5' long, soft cap at 3'.
    unseen_together = n_observed_total == 0
    n_observed_same[unseen_together] = 0
    n_observed_total[unseen_together] = 1

    p_same_partition = n_observed_same / n_observed_total
    p_one = p_same_partition == 1
    weights = np.where(~p_one, p_same_partition / (1 - p_same_partition), VERY_BIG)

    # [Sum, max] for each partitioning in S: (sum = Z, max = solution)
    #   Product for each partition in the partitioning:
    #       Product for each unique pair of elements in the partition:
    #           weights[i, j]

    # To find the max for a set S:
    #   Arbitrarily choose an element k from S
    #   For each possible subset T of S \ {k}:
    #       T = T | {k}
    #       Compute the product for T
    #       Multiply by the biggest product of S \ T
    #       Store the biggest

    def weight_product(subset: Iterable[FIRSTID]) -> float:
        p = 1
        for i, j in itertools.combinations(subset, 2):
            p *= weights[name_to_index[i], name_to_index[j]]
        return p

    @dataclasses.dataclass
    class PartitionProduct:
        maximum_product: float
        partition_function: float
        best_partition: frozenset[frozenset[FIRSTID]]

    @functools.cache
    def compute_partition_product(s: frozenset[FIRSTID]) -> PartitionProduct:
        if not s:
            return PartitionProduct(1, 1, frozenset())

        max_product = 0
        sum_product = 0
        maximiser = frozenset()
        element = min(s)
        for subset in powerset(s - {element}):
            subset = subset | {element}
            pp = compute_partition_product(s - subset)
            wp = weight_product(subset)
            product = wp * pp.maximum_product
            sum_product += wp * pp.partition_function
            if product > max_product:
                max_product = product
                maximiser = pp.best_partition | {subset}

        if not maximiser:
            raise RuntimeError()

        return PartitionProduct(max_product, sum_product, maximiser)

    elements = frozenset(all_names)
    pp = compute_partition_product(elements)

    block_probabilities = {}
    for block in pp.best_partition:
        block_probabilities[block] = (
            weight_product(block)
            * compute_partition_product(elements - block).partition_function
            / pp.partition_function
        )

    return Aggregation(
        aggregation=pp.best_partition,
        scores=block_probabilities,
        probability=pp.maximum_product / pp.partition_function,
    )


def aggregate_subset_ensembles(classifications: Sequence[Classification]):
    """Aggregates a subset of classifications using cluster ensembles.

    The subset should be a clique, i.e. these classifications
    should all be (transitively) in the same subjects. That's not
    a hard constraint but the bigger the size of the problem the
    worse it'll be computationally.

    Args:
        classifications: Subset of classifications.

    Returns:
        Consensus.
    """
    # Convert FIRST IDs into indices.
    all_names = sorted(get_all_first_ids(classifications))
    name_to_index = {name: i for i, name in enumerate(all_names)}

    labellings = []
    # Convert classifications into labellings.
    for c in classifications:
        labelling = np.zeros(len(all_names))
        for i, (_, radios) in enumerate(c.coord_matches):
            for r in radios:
                labelling[name_to_index[r]] = i
        labellings.append(labelling)
    labellings = np.array(labellings)

    consensus = cluster_ensembles.cluster_ensembles(labellings)
    clusters = max(consensus) + 1
    results = []
    for i in range(clusters):
        in_cluster = (consensus == i).nonzero()[0]
        results.append([all_names[j] for j in in_cluster])

    return results
