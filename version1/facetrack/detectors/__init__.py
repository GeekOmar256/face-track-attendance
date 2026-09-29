"""Detector registry, so scripts can take a plain ``--detector`` name."""

from __future__ import annotations

from typing import Any

from .base import Detection, FaceDetector
from .haar import HaarDetector
from .yunet import YuNetDetector

DETECTOR_NAMES = ("haar", "yunet")

__all__ = [
    "Detection",
    "FaceDetector",
    "HaarDetector",
    "YuNetDetector",
    "DETECTOR_NAMES",
    "get_detector",
]


def get_detector(name: str, **kwargs: Any) -> FaceDetector:
    key = (name or "").strip().lower()
    if key == "haar":
        return HaarDetector(**kwargs)
    if key == "yunet":
        return YuNetDetector(**kwargs)
    raise ValueError(f"unknown detector '{name}'. Choose one of: {', '.join(DETECTOR_NAMES)}")
