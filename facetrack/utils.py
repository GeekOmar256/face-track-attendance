"""Small helpers shared by the scripts: timing, image quality, dataset listing."""

from __future__ import annotations

import csv
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from . import config


class FPSMeter:
    """Frames per second over a sliding window, so the number on screen is steady."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque = deque(maxlen=window)
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            self._times.append(now - self._last)
        self._last = now
        return self.fps

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        mean = sum(self._times) / len(self._times)
        return 1.0 / mean if mean > 0 else 0.0


class Stopwatch:
    """Context manager that records elapsed milliseconds.

    Used for the per-frame detection timings the report needs for NFReq-3.
    """

    def __init__(self) -> None:
        self.ms = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0


def blur_score(image: np.ndarray) -> float:
    """Variance of the Laplacian. Higher means sharper.

    The dataset capture script uses this to reject soft frames, which is what
    keeps a self-captured dataset from being thirty copies of the same blur.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def slugify(text: str) -> str:
    """Turn a person's name into something safe for a folder name."""
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    return re.sub(r"[\s-]+", "_", cleaned)


def person_dir_name(student_id: str, name: str) -> str:
    return f"{slugify(str(student_id))}_{slugify(name)}"


def split_person_dir(dir_name: str) -> Tuple[str, str]:
    """Split ``210001_Student_Name`` into the id and the readable name."""
    parts = dir_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[0], parts[1].replace("_", " ")
    return "", dir_name.replace("_", " ")


def list_dataset(dataset_dir: Path = None) -> Dict[str, List[Path]]:
    """Map each person folder name to its sorted list of image paths."""
    root = Path(dataset_dir or config.DATASET_DIR)
    people: Dict[str, List[Path]] = {}
    if not root.exists():
        return people
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            f for f in person.iterdir()
            if f.suffix.lower() in config.IMAGE_EXTENSIONS
        )
        if images:
            people[person.name] = images
    return people


def iter_images(paths: Iterable[Path]):
    """Yield (path, bgr_image) skipping anything OpenCV refuses to decode."""
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  warning: could not read {path}, skipped")
            continue
        yield path, image


def largest_detection(detections: Sequence):
    """Pick the biggest face in the frame.

    Dataset images are captured one person at a time, so the largest box is the
    subject and anything else is a background face or a false positive.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d.w * d.h)


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[dict], fieldnames: Sequence[str] = None) -> str:
    """Render rows as a markdown table, ready to paste into the report."""
    if not rows:
        return ""
    fieldnames = list(fieldnames or rows[0].keys())
    lines = ["| " + " | ".join(fieldnames) + " |",
             "| " + " | ".join("---" for _ in fieldnames) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fieldnames) + " |")
    return "\n".join(lines)


def save_table(path_stem: Path, rows: Sequence[dict], fieldnames: Sequence[str] = None) -> None:
    """Write the same table as a CSV and as a markdown file."""
    write_csv(path_stem.with_suffix(".csv"), rows, fieldnames)
    text = markdown_table(rows, fieldnames)
    path_stem.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    print(f"\n{text}\n")
    print(f"written: {path_stem.with_suffix('.csv')}")
    print(f"written: {path_stem.with_suffix('.md')}")
