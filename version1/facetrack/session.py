"""A running pipeline plus the statistics it accumulates.

Both front ends, the web page and the desktop window, drive this class. The
processing happens on a background thread and the annotated frame is published
under a lock, so a user interface only ever reads the latest result and never
blocks the camera.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .camera import CameraSource
from .detectors import get_detector
from .draw import AMBER, GREEN, RED, draw_banner, draw_detections
from .recognizers import default_model_path, get_recognizer
from .utils import Stopwatch, split_person_dir


# ====================================================================== stats
class Stats:
    """The figures both interfaces display and export."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.frames = 0
            self.frames_with_face = 0
            self.total_faces = 0
            self.detect_ms: List[float] = []
            self.pipeline_ms: List[float] = []
            self.identities: Counter = Counter()
            self.unknown = 0
            self.started = time.time()

    def add(self, faces: int, detect_ms: float, pipeline_ms: float,
            labels: List[Tuple[str, bool]]) -> None:
        with self.lock:
            self.frames += 1
            self.total_faces += faces
            if faces:
                self.frames_with_face += 1
            self.detect_ms.append(detect_ms)
            self.pipeline_ms.append(pipeline_ms)
            for label, known in labels:
                if known:
                    self.identities[label] += 1
                else:
                    self.unknown += 1

    def snapshot(self) -> Dict:
        """A consistent copy of everything, safe to hand to a UI thread."""
        with self.lock:
            frames = self.frames
            detect = list(self.detect_ms)
            pipeline = list(self.pipeline_ms)
            identities = self.identities.most_common()
            unknown = self.unknown
            faces = self.total_faces
            with_face = self.frames_with_face
            elapsed = time.time() - self.started

        mean_pipeline = statistics.fmean(pipeline) if pipeline else 0.0
        people = []
        for label, count in identities:
            student_id, name = split_person_dir(label)
            people.append({
                "label": label,
                "id": student_id,
                "name": name,
                "frames": count,
                "share": round(100 * count / frames, 1) if frames else 0.0,
            })

        return {
            "frames": frames,
            "frames_with_face": with_face,
            "total_faces": faces,
            "detection_rate": round(100 * with_face / frames, 1) if frames else 0.0,
            "mean_detect_ms": round(statistics.fmean(detect), 1) if detect else 0.0,
            "mean_pipeline_ms": round(mean_pipeline, 1),
            "median_pipeline_ms": round(statistics.median(pipeline), 1) if pipeline else 0.0,
            "max_pipeline_ms": round(max(pipeline), 1) if pipeline else 0.0,
            "throughput_fps": round(1000 / mean_pipeline, 1) if mean_pipeline else 0.0,
            # NFReq-3 in the FYP1 plan allows one second of processing per frame.
            "meets_nfreq3": bool(pipeline) and mean_pipeline <= 1000,
            "unknown": unknown,
            "people": people,
            "elapsed_s": round(elapsed, 1),
        }

    def export_rows(self, detector: str, recognizer: str, source: str) -> List[dict]:
        data = self.snapshot()
        rows = [{
            "Source": source,
            "Detector": detector,
            "Recognizer": recognizer,
            "Frames processed": data["frames"],
            "Frames with a face": data["frames_with_face"],
            "Detection rate (%)": data["detection_rate"],
            "Total faces": data["total_faces"],
            "Mean detection (ms)": data["mean_detect_ms"],
            "Mean pipeline (ms)": data["mean_pipeline_ms"],
            "Median pipeline (ms)": data["median_pipeline_ms"],
            "Max pipeline (ms)": data["max_pipeline_ms"],
            "Throughput (fps)": data["throughput_fps"],
            "NFReq-3 (<=1000 ms)": "PASS" if data["meets_nfreq3"] else "FAIL",
            "Unknown faces": data["unknown"],
            "Identified person": "",
            "Student ID": "",
            "Frames identified": "",
        }]
        for person in data["people"]:
            rows.append({
                "Source": source,
                "Detector": detector,
                "Recognizer": recognizer,
                "Frames processed": "",
                "Frames with a face": "",
                "Detection rate (%)": "",
                "Total faces": "",
                "Mean detection (ms)": "",
                "Mean pipeline (ms)": "",
                "Median pipeline (ms)": "",
                "Max pipeline (ms)": "",
                "Throughput (fps)": "",
                "NFReq-3 (<=1000 ms)": "",
                "Unknown faces": "",
                "Identified person": person["name"],
                "Student ID": person["id"],
                "Frames identified": person["frames"],
            })
        return rows


# ==================================================================== session
class PipelineSession:
    """Owns the camera, the detector, the recognizer and the worker thread."""

    def __init__(self) -> None:
        self.stats = Stats()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self.state = "idle"                 # idle | running | finished | error
        self.error: Optional[str] = None
        self.detector_name = "-"
        self.recognizer_name = "none"
        self.source_label = "-"
        self.detector_description = ""
        self._source_spec: Tuple[Optional[str], int] = (None, config.CAMERA_INDEX)

    # ------------------------------------------------------------ lifecycle
    def start(self, source: Optional[str] = None, camera_index: int = config.CAMERA_INDEX,
              detector: str = "haar", recognizer: str = "none",
              threshold: Optional[float] = None, loop: bool = False,
              overlay: bool = True, **detector_kwargs) -> None:
        """Build the pipeline and begin processing. Raises on a bad configuration."""
        if self.state == "running":
            raise RuntimeError("a session is already running")

        # Remembered so a finished video or photo folder can be reopened when
        # looping is on.
        self._source_spec = (source, camera_index)
        camera = self._open_camera()

        try:
            det = self._build_detector(detector, detector_kwargs)
            rec = self._build_recognizer(recognizer, threshold)
        except Exception:
            camera.release()
            raise

        self.detector_name = det.name
        self.detector_description = det.describe()
        self.recognizer_name = rec.name if rec else "none"
        self.error = None
        self.state = "running"
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(camera, det, rec, loop, overlay), daemon=True)
        self._thread.start()

    def _open_camera(self) -> CameraSource:
        source, camera_index = self._source_spec
        if source:
            self.source_label = Path(source).name
            return CameraSource(source=source)
        self.source_label = f"camera {camera_index}"
        return CameraSource(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT,
                            camera_index=camera_index)

    @staticmethod
    def _build_detector(name: str, kwargs: dict):
        if name == "haar":
            return get_detector(
                "haar",
                scale_factor=float(kwargs.get("scale_factor", config.HAAR_SCALE_FACTOR)),
                min_neighbors=int(kwargs.get("min_neighbors", config.HAAR_MIN_NEIGHBORS)),
                min_size=(int(kwargs.get("min_size", config.HAAR_MIN_SIZE[0])),) * 2,
            )
        return get_detector(
            "yunet",
            score_threshold=float(kwargs.get("score_threshold", config.YUNET_SCORE_THRESHOLD)),
        )

    @staticmethod
    def _build_recognizer(name: str, threshold: Optional[float]):
        if not name or name == "none":
            return None
        kwargs = {} if threshold is None else {"threshold": float(threshold)}
        recognizer = get_recognizer(name, **kwargs)
        recognizer.load(default_model_path(name))
        return recognizer

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None
        if self.state == "running":
            self.state = "idle"

    # ---------------------------------------------------------------- worker
    def _run(self, camera, detector, recognizer, loop: bool, overlay: bool) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = camera.read()
                if not ok or frame is None:
                    if loop and camera.kind in ("video", "images", "image"):
                        camera.release()
                        camera = self._open_camera()
                        continue
                    self.state = "finished"
                    break

                with Stopwatch() as detect_watch:
                    detections = detector.detect(frame)

                results = []
                with Stopwatch() as recog_watch:
                    if recognizer is not None:
                        results = [recognizer.predict(frame, d) for d in detections]

                annotated, label_pairs = self._annotate(
                    frame, detections, results, detector, recognizer,
                    detect_watch.ms + recog_watch.ms, overlay)

                self.stats.add(len(detections), detect_watch.ms,
                               detect_watch.ms + recog_watch.ms, label_pairs)

                with self._frame_lock:
                    self._frame = annotated
                    self._frame_id += 1

                if camera.kind in ("image", "images"):
                    # Still photographs do not need to be reprocessed at video rate.
                    time.sleep(0.15)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
        finally:
            try:
                camera.release()
            except Exception:
                pass

    @staticmethod
    def _annotate(frame, detections, results, detector, recognizer, pipeline_ms, overlay):
        canvas = frame.copy()
        labels, colours, pairs = [], [], []
        for i, detection in enumerate(detections):
            if results:
                result = results[i]
                _, name = split_person_dir(result.label)
                if result.is_known:
                    labels.append(f"{name} ({result.score:.2f})")
                    colours.append(GREEN)
                    pairs.append((result.label, True))
                else:
                    labels.append(config.UNKNOWN_LABEL)
                    colours.append(RED)
                    pairs.append((result.label, False))
            else:
                labels.append(f"{detection.score:.2f}")
                colours.append(AMBER)

        draw_detections(canvas, detections, labels=labels, colors=colours)
        if overlay:
            lines = [f"{detector.name}" + (f" + {recognizer.name}" if recognizer else ""),
                     f"faces: {len(detections)}",
                     f"pipeline: {pipeline_ms:.1f} ms"]
            draw_banner(canvas, lines)
        return canvas, pairs

    # ---------------------------------------------------------------- output
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def frame_id(self) -> int:
        with self._frame_lock:
            return self._frame_id

    def latest_jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame = self.latest_frame()
        if frame is None:
            return None
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes() if ok else None

    def save_snapshot(self) -> Optional[Path]:
        frame = self.latest_frame()
        if frame is None:
            return None
        config.ensure_dirs()
        folder = config.OUTPUT_DIR / "snapshots"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    def status(self) -> Dict:
        return {
            "state": self.state,
            "error": self.error,
            "detector": self.detector_name,
            "detector_description": self.detector_description,
            "recognizer": self.recognizer_name,
            "source": self.source_label,
            "stats": self.stats.snapshot(),
        }
