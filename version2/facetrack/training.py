"""Building a recognizer from the dataset.

Version 2 lifts this out of scripts/train.py so the browser can retrain after an
enrollment without anyone opening a terminal. The command line script now calls
the same functions, so both paths train identically.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from . import config
from .recognizers import TrainingSample, default_model_path, get_recognizer
from .utils import list_dataset, split_person_dir


def collect_samples(dataset_dir: Path, detector,
                    on_person: Optional[Callable[[dict], None]] = None
                    ) -> Tuple[List[TrainingSample], List[dict]]:
    """Detect one face per dataset image and return the training samples."""
    people = list_dataset(dataset_dir)
    if not people:
        raise ValueError(f"no enrolled people found in {dataset_dir}")

    samples: List[TrainingSample] = []
    report: List[dict] = []
    for person, paths in people.items():
        found = 0
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            detection = detector.detect_largest(image)
            if detection is None:
                continue
            samples.append(TrainingSample(label=person, image=image, detection=detection))
            found += 1

        student_id, name = split_person_dir(person)
        entry = {
            "folder": person,
            "id": student_id,
            "name": name,
            "images": len(paths),
            "faces_found": found,
            "rate": round(100 * found / len(paths), 1) if paths else 0.0,
        }
        report.append(entry)
        if on_person is not None:
            on_person(entry)

    return samples, report


def train_recognizer(recognizer_name: str, detector, dataset_dir: Optional[Path] = None,
                     model_path: Optional[Path] = None,
                     on_person: Optional[Callable[[dict], None]] = None) -> Dict:
    """Train one recognizer and save it. Returns a summary for the caller to show."""
    dataset_dir = Path(dataset_dir or config.DATASET_DIR)
    samples, report = collect_samples(dataset_dir, detector, on_person)

    people = sorted({s.label for s in samples})
    if len(people) < 2 and recognizer_name == "lbph":
        raise ValueError(
            f"LBPH needs at least two enrolled people, found {len(people)}. "
            "Enroll another person first."
        )
    if not samples:
        raise ValueError("no usable faces were found in the dataset")

    recognizer = get_recognizer(recognizer_name)
    recognizer.fit(samples)
    path = Path(model_path or default_model_path(recognizer_name))
    recognizer.save(path)

    return {
        "recognizer": recognizer_name,
        "detector": detector.name,
        "people": len(people),
        "faces": len(samples),
        "model": str(path),
        "report": report,
    }


class TrainingJob:
    """Runs training on a background thread so the web request returns at once."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self.state = "idle"            # idle | running | done | error
        self.error: Optional[str] = None
        self.message = ""
        self.result: Optional[Dict] = None
        self.progress: List[dict] = []

    @property
    def running(self) -> bool:
        return self.state == "running"

    def start(self, recognizer_name: str, detector) -> None:
        if self.running:
            raise RuntimeError("training is already running")
        self.state = "running"
        self.error = None
        self.result = None
        self.progress = []
        self.message = f"training {recognizer_name} using {detector.name} detections"
        self._thread = threading.Thread(
            target=self._run, args=(recognizer_name, detector), daemon=True)
        self._thread.start()

    def _run(self, recognizer_name: str, detector) -> None:
        try:
            result = train_recognizer(
                recognizer_name, detector, on_person=self.progress.append)
            self.result = result
            self.message = (f"trained {result['recognizer']} on {result['faces']} faces "
                            f"from {result['people']} people")
            self.state = "done"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.message = self.error
            self.state = "error"

    def status(self) -> Dict:
        return {
            "state": self.state,
            "error": self.error,
            "message": self.message,
            "result": self.result,
            "progress": self.progress,
        }
