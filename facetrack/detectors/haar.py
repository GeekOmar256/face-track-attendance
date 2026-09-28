"""Haar Cascade face detector: the baseline the advisor asked the team to start from.

This is the Viola and Jones classifier discussed in the FYP1 literature review.
The cascade XML ships inside OpenCV, so nothing has to be downloaded for the
baseline to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .. import config
from .base import Detection, FaceDetector


class HaarDetector(FaceDetector):
    name = "haar"

    def __init__(
        self,
        cascade_name: str = config.HAAR_CASCADE_NAME,
        scale_factor: float = config.HAAR_SCALE_FACTOR,
        min_neighbors: int = config.HAAR_MIN_NEIGHBORS,
        min_size: Tuple[int, int] = config.HAAR_MIN_SIZE,
        equalize: bool = True,
    ) -> None:
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = tuple(min_size)
        self.equalize = equalize

        cascade_path = self._resolve_cascade(cascade_name)
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade from {cascade_path}")
        self.cascade_path = cascade_path

    @staticmethod
    def _resolve_cascade(cascade_name: str) -> Path:
        candidate = Path(cascade_name)
        if candidate.is_file():
            return candidate
        bundled = Path(cv2.data.haarcascades) / cascade_name
        if bundled.is_file():
            return bundled
        raise FileNotFoundError(
            f"cascade '{cascade_name}' not found as a path or inside {cv2.data.haarcascades}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.equalize:
            # Histogram equalisation is the cheapest defence against the lighting
            # sensitivity the FYP1 literature review attributes to Haar.
            gray = cv2.equalizeHist(gray)

        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Haar gives no confidence value, so every detection is reported at 1.0.
        # That difference from YuNet is worth a sentence in the comparison.
        return [
            Detection(int(x), int(y), int(w), int(h), score=1.0)
            for (x, y, w, h) in boxes
        ]

    def describe(self) -> str:
        return (
            f"Haar Cascade ({self.cascade_path.name}, scaleFactor={self.scale_factor}, "
            f"minNeighbors={self.min_neighbors}, minSize={self.min_size})"
        )
