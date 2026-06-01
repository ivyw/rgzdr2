"""Other aggregation strategies for classifications."""

from collections.abc import Sequence

import ClusterEnsembles as cluster_ensembles
import numpy as np

from rgz.classifications import Classification


def aggregate_subset(classifications: Sequence[Classification]):
    """Aggregates a subset of classifications.

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
    all_names = set()
    for c in classifications:
        for _, radios in c.coord_matches:
            for r in radios:
                all_names.add(r)
    all_names = sorted(all_names)
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
    print(consensus)
    clusters = max(consensus) + 1
    results = []
    for i in range(clusters):
        in_cluster = (consensus == i).nonzero()[0]
        results.append([all_names[j] for j in in_cluster])

    return results
