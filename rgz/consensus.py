"""Aggregates classifications."""

import logging
from pathlib import Path
from typing import Self

import attr

from rgz import classifications
from rgz import rgz
from rgz import subjects

logger = logging.getLogger(__name__)


@attr.s
class ConsensusSource:
    """A radio source with a cross-identified IR host galaxy.

    Attributes:
        zid: Zooniverse ID of the subject containing this source.
        components: FIRST components of the source.
        host_name: Name of the AllWISE host if it exists.
        radio_agreement: Number of citizen scientists who agreed
            with the radio component combination.
        ir_agreement: Number of citizen scientists who agreed that
            the IR host was an IR host of any radio source. You might
            want to use joint_agreement instead of this.
        joint_agreement: Number of citizen scientists who agreed with
            the IR host galaxy _and_ the radio component combination.
        votes: Number of citizen scientists who voted on this subject.
    """

    zid: subjects.ZooniverseID = attr.ib()
    components: set[subjects.FIRSTID] = attr.ib()
    host_name: classifications.ALLWISEID | None = attr.ib()
    radio_agreement: int = attr.ib()
    ir_agreement: int = attr.ib()
    joint_agreement: int = attr.ib()
    votes: int = attr.ib()

    @classmethod
    def from_json(cls, obj: rgz.JSON) -> Self:
        """Reads a JSON dict."""
        return cls(
            zid=obj["zid"],
            components=set(obj["components"]),
            host_name=obj["host_name"],
            radio_agreement=obj["radio_agreement"],
            ir_agreement=obj["ir_agreement"],
            joint_agreement=obj["joint_agreement"],
            votes=obj["votes"],
        )

    def to_json(self) -> rgz.JSON:
        """Converts the ConsensusSource to a JSON-compatible dict."""
        return {
            "zid": self.zid,
            "components": sorted(self.components),
            "host_name": self.host_name,
            "radio_agreement": self.radio_agreement,
            "ir_agreement": self.ir_agreement,
            "joint_agreement": self.joint_agreement,
            "votes": self.votes,
        }


def aggregate(subjects_path: Path, classifications_path: Path, out_path: Path) -> None:
    """Aggregates classifications into a consensus for each subject.

    Args:
        subjects_path: Path to reduced subjects JSON.
        classifications_path: Path to reduced, cross-matched subjects JSON.
        out_path: Path to output the consensus JSON.
    """
    raise NotImplementedError()
