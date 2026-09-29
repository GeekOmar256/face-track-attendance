"""The contract every detector satisfies.

Keeping this interface narrow is what lets the advisor's comparison between
Haar Cascade and a deep neural network detector be a command line flag rather
than a second copy of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """One detected face.

    ``landmarks`` and ``raw`` are filled in by YuNet only. Haar Cascade reports
    a bounding box and nothing else, which is exactly why SFace alignment is
    weaker behind Haar. See the README.
    """

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    landmarks: Optional[np.ndarray] = None   # (5, 2) eye, eye, nose, mouth, mouth
    raw: Optional[np.ndarray] = None         # YuNet's 15 value row, needed by alignCrop

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def crop(
        self,
        frame: np.ndarray,
        margin: float = 0.0,
        size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Cut the face out of the frame, optionally padded and resized.

        The margin matters for recognition: a box cut tight to the Haar result
        clips the chin and hairline, and both recognizers do better with a
        little context around the face.
        """
        height, width = frame.shape[:2]
        mx = int(round(self.w * margin))
        my = int(round(self.h * margin))
        x1 = max(0, self.x - mx)
        y1 = max(0, self.y - my)
        x2 = min(width, self.x + self.w + mx)
        y2 = min(height, self.y + self.h + my)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        roi = frame[y1:y2, x1:x2]
        if size is not None and roi.size:
            roi = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        return roi


class FaceDetector(ABC):
    """Base class for the detectors."""

    name = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return every face found in a BGR frame."""

    def detect_largest(self, frame: np.ndarray) -> Optional[Detection]:
        detections = self.detect(frame)
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)

    def describe(self) -> str:
        """One line describing the configuration, printed by the scripts."""
        return self.name
