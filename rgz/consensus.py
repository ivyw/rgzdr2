"""Aggregates classifications."""

import collections
from collections.abc import Iterable
import json
import logging
from pathlib import Path
from typing import Self

import attr

from rgz import classifications
from rgz import rgz
from rgz import subjects

logger = logging.getLogger(__name__)

_JSON_INDENT = 2


@attr.s
class ConsensusSource:
    """A radio source with a cross-identified IR host galaxy.

    Attributes:
        zid: Zooniverse ID of the subject containing this source.
        components: FIRST components of the source.
        host_name: Name of the AllWISE host if it exists.
        n_radio_agreement: Number of citizen scientists who agreed
            with the radio component combination.
        n_ir_agreement: Number of citizen scientists who agreed that
            the IR host was an IR host of any radio source. You might
            want to use n_joint_agreement instead of this.
        n_joint_agreement: Number of citizen scientists who agreed with
            the IR host galaxy _and_ the radio component combination.
        votes: Number of citizen scientists who voted on this subject.
    """

    zid: subjects.ZooniverseID = attr.ib()
    components: set[subjects.FIRSTID] = attr.ib()
    host_name: classifications.ALLWISEID | None = attr.ib()
    n_radio_agreement: int = attr.ib()
    n_ir_agreement: int = attr.ib()
    n_joint_agreement: int = attr.ib()
    n_votes: int = attr.ib()

    @classmethod
    def from_json(cls, obj: rgz.JSON) -> Self:
        """Reads a JSON dict."""
        return cls(
            zid=obj["zid"],
            components=set(obj["components"]),
            host_name=obj["host_name"],
            n_radio_agreement=obj["n_radio_agreement"],
            n_ir_agreement=obj["n_ir_agreement"],
            n_joint_agreement=obj["n_joint_agreement"],
            n_votes=obj["votes"],
        )

    def to_json(self) -> rgz.JSON:
        """Converts the ConsensusSource to a JSON-compatible dict."""
        return {
            "zid": self.zid,
            "components": sorted(self.components),
            "host_name": self.host_name,
            "n_radio_agreement": self.n_radio_agreement,
            "n_ir_agreement": self.n_ir_agreement,
            "n_joint_agreement": self.n_joint_agreement,
            "votes": self.n_votes,
        }


def aggregate_subject(
    subject: subjects.Subject, classifications: list[classifications.Classification]
) -> list[ConsensusSource]:
    """Aggregates classifications into consensus sources for a single subject."""
    # Represent each combination of radio objects by something deterministic and hashable.
    radio_combinations = [cl.radio_combinations() for cl in classifications]

    # What's the most common combination (consensus)?
    counter = collections.Counter(radio_combinations)
    ((consensus_radio, consensus_radio_count),) = counter.most_common(1)
    consensus_radio_pc = consensus_radio_count / len(radio_combinations)

    # Amongst people who chose this, what IR was most common?
    source_to_ir_options = {source: [] for source in consensus_radio.sources()}
    for ident, cl in zip(radio_combinations, classifications):
        if ident == consensus_radio:
            for ir, radios in cl.ir_matches:
                source_to_ir_options[radios].append(ir)

    # How many people thought _each_ IR object was a host, regardless
    # of their choice of radio?
    n_ir_votes = collections.Counter()
    for classification in classifications:
        for ir, _ in classification.ir_matches:
            n_ir_votes[ir] += 1

    matches = []
    for first, irs in source_to_ir_options.items():
        ((consensus_ir, consensus_ir_count),) = collections.Counter(irs).most_common(1)
        matches.append(
            ConsensusSource(
                zid=classifications[0].zid,
                components=set(first),
                host_name=consensus_ir,
                n_joint_agreement=consensus_ir_count,
                n_ir_agreement=n_ir_votes[consensus_ir],
                n_radio_agreement=consensus_radio_count,
                n_votes=len(classifications),
            )
        )

    return matches


def aggregate(subjects_path: Path, classifications_path: Path, out_path: Path) -> None:
    """Aggregates classifications into a consensus for each subject.

    Args:
        subjects_path: Path to reduced subjects JSON.
        classifications_path: Path to reduced, cross-matched subjects JSON.
        out_path: Path to output the consensus JSON.
    """
    with open(subjects_path) as f:
        all_subjects = [subjects.Subject.from_json(j) for j in json.load(f)]

    with open(classifications_path) as f:
        all_classifications = [
            classifications.Classification.from_json(j) for j in json.load(f)
        ]

    zid_to_subject = {s.zid: s for s in all_subjects}

    zid_to_classifications = collections.defaultdict(list)
    for classification in all_classifications:
        zid_to_classifications[classification.zid].append(classification)

    consensuses = []
    for zid, zid_classifications in zid_to_classifications.items():
        subject = zid_to_subject[zid]
        consensuses.extend(aggregate_subject(subject, zid_classifications))

    # TODO: Sort the output for reproducibility.
    with open(out_path, "w") as f:
        json.dump([c.to_json() for c in consensuses], f, indent=_JSON_INDENT)
