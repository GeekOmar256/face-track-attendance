"""The contract every recognizer satisfies.

Both recognizers report the best matching identity together with the raw score
that produced it, and separately say whether that score cleared the decision
threshold. Keeping those two apart is what lets evaluate_recognition.py sweep
the threshold after the fact instead of re-running the whole pipeline for every
candidate value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .. import config
from ..detectors.base import Detection


@dataclass
class TrainingSample:
    """One enrolled face: the image it came from and where the face sits in it."""

    label: str
    image: np.ndarray
    detection: Detection


@dataclass
class RecognitionResult:
    label: str          # closest identity in the gallery, whatever the threshold says
    score: float        # raw distance or similarity, as the recognizer defines it
    is_known: bool      # did the score clear the threshold

    @property
    def display_label(self) -> str:
        return self.label if self.is_known else config.UNKNOWN_LABEL


class FaceRecognizer(ABC):
    """Base class for the recognizers."""

    name = "base"
    higher_is_better = True     # SFace similarity yes, LBPH distance no

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self._labels: List[str] = []

    # ------------------------------------------------------------ training
    @abstractmethod
    def fit(self, samples: Sequence[TrainingSample]) -> None:
        """Build the model or gallery from enrolled faces."""

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> None:
        """Persist the trained model."""

    @abstractmethod
    def load(self, path: Optional[Path] = None) -> None:
        """Restore a previously trained model."""

    # ------------------------------------------------------------ inference
    @abstractmethod
    def predict(self, frame: np.ndarray, detection: Detection) -> RecognitionResult:
        """Identify one detected face."""

    # ------------------------------------------------------------ helpers
    def passes(self, score: float) -> bool:
        """Does this score clear the threshold, in whichever direction counts."""
        if self.higher_is_better:
            return score >= self.threshold
        return score <= self.threshold

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    def describe(self) -> str:
        direction = "higher is better" if self.higher_is_better else "lower is better"
        return (
            f"{self.name} ({len(self._labels)} enrolled, "
            f"threshold={self.threshold:g}, {direction})"
        )

    @staticmethod
    def _unknown(score: float) -> RecognitionResult:
        return RecognitionResult(label=config.UNKNOWN_LABEL, score=score, is_known=False)
