"""Recognizer registry, so scripts can take a plain ``--recognizer`` name."""

from __future__ import annotations

from typing import Any

from .base import FaceRecognizer, RecognitionResult, TrainingSample
from .lbph import LBPHRecognizer
from .sface import SFaceRecognizer

RECOGNIZER_NAMES = ("lbph", "sface")

__all__ = [
    "FaceRecognizer",
    "RecognitionResult",
    "TrainingSample",
    "LBPHRecognizer",
    "SFaceRecognizer",
    "RECOGNIZER_NAMES",
    "get_recognizer",
    "default_model_path",
]


def get_recognizer(name: str, **kwargs: Any) -> FaceRecognizer:
    key = (name or "").strip().lower()
    if key == "lbph":
        return LBPHRecognizer(**kwargs)
    if key == "sface":
        return SFaceRecognizer(**kwargs)
    raise ValueError(
        f"unknown recognizer '{name}'. Choose one of: {', '.join(RECOGNIZER_NAMES)}"
    )


def default_model_path(name: str):
    from .. import config

    key = (name or "").strip().lower()
    if key == "lbph":
        return config.LBPH_MODEL
    if key == "sface":
        return config.SFACE_GALLERY
    raise ValueError(f"unknown recognizer '{name}'")
