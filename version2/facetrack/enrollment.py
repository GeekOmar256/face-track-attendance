"""Enrolling a new person: their details, and capturing their face.

Version 2 adds this. In version 1 a person could only be enrolled from the
command line with capture_dataset.py, and the only thing recorded about them
was whatever was encoded in the folder name. Here the details are stored
properly in a person.json beside the images, so the interface can show a real
record rather than a parsed folder name.

The capture gate is the same one the command line tool uses: a frame is kept
only when exactly one face is visible and the crop is sharp enough, with a gap
between saves. Without it a run produces a burst of near identical blurred
frames and any accuracy measured on them is meaningless.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import config
from .camera import CameraSource
from .detectors import get_detector
from .draw import AMBER, GREEN, RED, draw_banner, draw_detections
from .framesource import FrameSource
from .utils import blur_score, person_dir_name, split_person_dir

PERSON_FILE = "person.json"

# Fields kept for each enrolled person. Only the first two are required; the
# rest are there because an attendance record is more useful than a name alone.
DETAIL_FIELDS = ("student_id", "name", "email", "programme", "section", "notes")

# How much bigger the main face must be than the next one before the others are
# treated as noise. Haar in particular reports spurious boxes on hair, beards and
# patterned backgrounds, and a portrait where one face dwarfs the rest is one
# person photographed badly, not a group. Two faces of similar size really are
# two people, and filing that under one name would mislabel somebody.
DOMINANT_RATIO = 2.0

POSE_PROMPTS = [
    "look straight at the camera",
    "turn slightly left",
    "turn slightly right",
    "tilt your head a little",
    "smile",
    "neutral expression again",
]


# ================================================================== records
def person_folder(student_id: str, name: str, dataset_dir: Optional[Path] = None) -> Path:
    root = Path(dataset_dir or config.DATASET_DIR)
    return root / person_dir_name(student_id, name)


def write_details(folder: Path, details: Dict) -> Dict:
    """Store the person's record next to their images."""
    folder.mkdir(parents=True, exist_ok=True)
    record = {field: str(details.get(field, "") or "").strip() for field in DETAIL_FIELDS}
    existing = read_details(folder)
    record["enrolled_at"] = existing.get("enrolled_at") or time.strftime("%Y-%m-%d %H:%M:%S")
    record["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (folder / PERSON_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def read_details(folder: Path) -> Dict:
    path = Path(folder) / PERSON_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def count_images(folder: Path) -> int:
    return sum(1 for f in Path(folder).iterdir()
               if f.suffix.lower() in config.IMAGE_EXTENSIONS)


def list_people(dataset_dir: Optional[Path] = None) -> List[Dict]:
    """Every enrolled person, with their record and how many images they have.

    Folders created by the command line tool have no person.json, so their
    details are recovered from the folder name and reported as such.
    """
    root = Path(dataset_dir or config.DATASET_DIR)
    people: List[Dict] = []
    if not root.is_dir():
        return people

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        images = count_images(folder)
        if not images:
            continue
        details = read_details(folder)
        if not details:
            student_id, name = split_person_dir(folder.name)
            details = {"student_id": student_id, "name": name}
        entry = {field: details.get(field, "") for field in DETAIL_FIELDS}
        entry.update({
            "folder": folder.name,
            "images": images,
            "enrolled_at": details.get("enrolled_at", ""),
            "has_record": bool(read_details(folder)),
        })
        people.append(entry)
    return people


def pick_subject(detections: List, ratio: float = DOMINANT_RATIO):
    """Choose the one face an image should be filed under.

    Returns (detection, extra_count, reason). A reason means the image was
    refused: either nothing was found, or several faces are of comparable size
    and the picture holds more than one person.
    """
    if not detections:
        return None, 0, "no face found in this image"

    ordered = sorted(detections, key=lambda d: d.area, reverse=True)
    extra = len(ordered) - 1
    if extra and ordered[0].area < ordered[1].area * ratio:
        return None, extra, (f"{len(ordered)} faces of similar size, "
                             "one person per image")
    return ordered[0], extra, None


def next_image_index(folder: Path) -> int:
    """Continue numbering after whatever is already in the folder."""
    return count_images(folder) + 1


def add_photo(details: Dict, data: bytes, detector,
              preview: Optional[FrameSource] = None) -> Dict:
    """Add one supplied photograph to a person's folder.

    This is the alternative to capturing from the camera: the team may already
    have pictures of someone. The same rule applies as during capture, exactly
    one face per image, because an image holding two people cannot be filed
    under one name without mislabelling somebody.
    """
    student_id = str(details.get("student_id", "")).strip()
    name = str(details.get("name", "")).strip()
    if not student_id or not name:
        return {"ok": False, "reason": "a student ID and a name are both required"}

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"ok": False, "reason": "not a readable image"}

    detection, extra, reason = pick_subject(detector.detect(image))
    if reason:
        return {"ok": False, "reason": reason}

    folder = person_folder(student_id, name)
    folder.mkdir(parents=True, exist_ok=True)
    write_details(folder, details)

    path = folder / f"{student_id}_{next_image_index(folder):03d}.jpg"
    cv2.imwrite(str(path), image)

    if preview is not None:
        annotated = image.copy()
        draw_detections(annotated, [detection], colors=[GREEN])
        lines = [f"{name} ({student_id})", f"added {path.name}",
                 f"{count_images(folder)} images in total"]
        if extra:
            lines.append(f"ignored {extra} smaller detection(s)")
        draw_banner(annotated, lines)
        preview.publish(annotated)

    return {
        "ok": True,
        "file": path.name,
        "folder": folder.name,
        "images": count_images(folder),
        "extra_detections": extra,
        "sharpness": round(blur_score(detection.crop(image, margin=config.CAPTURE_MARGIN)), 1),
    }


def delete_person(folder_name: str, dataset_dir: Optional[Path] = None) -> bool:
    """Remove one person's folder. Refuses anything that is not a direct child."""
    root = Path(dataset_dir or config.DATASET_DIR).resolve()
    target = (root / folder_name).resolve()
    if target.parent != root or not target.is_dir():
        return False
    shutil.rmtree(target)
    return True


# ================================================================== capture
class EnrollmentSession(FrameSource):
    """Captures a set of images for one person, with a live preview."""

    def __init__(self) -> None:
        super().__init__()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.state = "idle"               # idle | running | done | error
        self.error: Optional[str] = None
        self.message = ""
        self.saved = 0
        self.target = 0
        self.rejected_blur = 0
        self.rejected_faces = 0
        self.folder: Optional[Path] = None
        self.details: Dict = {}

    # ------------------------------------------------------------ lifecycle
    def start(self, details: Dict, count: int = config.CAPTURE_DEFAULT_COUNT,
              detector: str = "haar", camera_index: int = config.CAMERA_INDEX,
              source: Optional[str] = None, blur_threshold: float = config.CAPTURE_BLUR_THRESHOLD,
              gap: int = config.CAPTURE_MIN_FRAME_GAP, replace: bool = False,
              rotate: int = config.CAMERA_ROTATION, **detector_kwargs) -> None:
        if self.state == "running":
            raise RuntimeError("an enrollment is already running")

        student_id = str(details.get("student_id", "")).strip()
        name = str(details.get("name", "")).strip()
        if not student_id or not name:
            raise ValueError("a student ID and a name are both required")

        folder = person_folder(student_id, name)
        folder.mkdir(parents=True, exist_ok=True)
        if replace:
            for existing in folder.iterdir():
                if existing.suffix.lower() in config.IMAGE_EXTENSIONS:
                    existing.unlink()

        self.details = write_details(folder, details)
        self.folder = folder
        self.saved = 0
        self.target = max(1, int(count))
        self.rejected_blur = 0
        self.rejected_faces = 0
        self.error = None
        self.message = "starting"
        self.state = "running"
        self._stop.clear()

        det = get_detector(
            detector,
            **({"scale_factor": float(detector_kwargs.get("scale_factor", config.HAAR_SCALE_FACTOR)),
                "min_neighbors": int(detector_kwargs.get("min_neighbors", config.HAAR_MIN_NEIGHBORS)),
                "min_size": (int(detector_kwargs.get("min_size", config.HAAR_MIN_SIZE[0])),) * 2}
               if detector == "haar" else
               {"score_threshold": float(detector_kwargs.get(
                   "score_threshold", config.YUNET_SCORE_THRESHOLD))}))

        camera = (CameraSource(source=source, rotate=rotate) if source else
                  CameraSource(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT,
                               camera_index=camera_index, rotate=rotate))

        self._thread = threading.Thread(
            target=self._run, args=(camera, det, blur_threshold, max(1, int(gap))),
            daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None
        if self.state == "running":
            self.state = "done" if self.saved else "idle"

    # --------------------------------------------------------------- worker
    def _run(self, camera, detector, blur_threshold: float, gap: int) -> None:
        since_save = 0
        start_index = self._next_index()
        try:
            while not self._stop.is_set() and self.saved < self.target:
                ok, frame = camera.read()
                if not ok or frame is None:
                    self.message = "the source ended before the target was reached"
                    break

                detections = detector.detect(frame)
                since_save += 1
                colour = RED

                detection, extra, reason = pick_subject(detections)
                if reason:
                    self.message = ("no face visible" if not detections
                                    else f"{len(detections)} faces, one person at a time")
                    self.rejected_faces += 1
                else:
                    crop = detection.crop(frame, margin=config.CAPTURE_MARGIN)
                    sharpness = blur_score(crop)
                    if sharpness < blur_threshold:
                        self.message = f"too blurred ({sharpness:.0f} < {blur_threshold:.0f})"
                        self.rejected_blur += 1
                        colour = AMBER
                    elif since_save < gap:
                        self.message = "hold still, next capture shortly"
                        colour = AMBER
                    else:
                        index = start_index + self.saved
                        path = self.folder / f"{self.details['student_id']}_{index:03d}.jpg"
                        # The full frame is saved, not the crop, so the detector
                        # comparison stays honest. See the README.
                        cv2.imwrite(str(path), frame)
                        self.saved += 1
                        since_save = 0
                        self.message = f"saved {path.name} (sharpness {sharpness:.0f})"
                        colour = GREEN

                preview = frame.copy()
                draw_detections(preview, detections, colors=[colour] * len(detections))
                draw_banner(preview, [
                    f"{self.details.get('name', '')} ({self.details.get('student_id', '')})",
                    f"captured: {self.saved} / {self.target}",
                    POSE_PROMPTS[(self.saved // 5) % len(POSE_PROMPTS)],
                    self.message,
                ])
                self.publish(preview)

            if self.saved >= self.target:
                self.message = f"captured {self.saved} images"
            self.state = "done"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.message = self.error
            self.state = "error"
        finally:
            try:
                camera.release()
            except Exception:
                pass

    def _next_index(self) -> int:
        """Continue numbering after any images already in the folder."""
        if self.folder is None:
            return 1
        existing = [f for f in self.folder.iterdir()
                    if f.suffix.lower() in config.IMAGE_EXTENSIONS]
        return len(existing) + 1

    # --------------------------------------------------------------- status
    def status(self) -> Dict:
        return {
            "state": self.state,
            "error": self.error,
            "message": self.message,
            "saved": self.saved,
            "target": self.target,
            "rejected_blur": self.rejected_blur,
            "rejected_faces": self.rejected_faces,
            "folder": self.folder.name if self.folder else "",
            "details": self.details,
            "images_in_folder": count_images(self.folder) if self.folder else 0,
        }
