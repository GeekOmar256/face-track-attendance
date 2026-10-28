import logging
import platform
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "data" / "models"


def jpeg(image, width=None):
    if width and image.shape[1] > width:
        image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Cannot encode image")
    return encoded.tobytes()


def decode(content):
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable image")
    if image.shape[0] * image.shape[1] > 25000000:
        raise ValueError("Image exceeds 25 megapixels")
    if image.shape[1] > 1280:
        image = cv2.resize(image, (1280, round(image.shape[0]*1280/image.shape[1])))
    return image


def crop(image, box):
    left, top, width, height = map(int, box)
    bottom, right = image.shape[:2]
    face = image[max(0, top):min(bottom, top+height), max(0, left):min(right, left+width)]
    if not face.size:
        raise ValueError("Empty face crop")
    return face


class Detector:
    def __init__(self, name, settings):
        self.name = name
        self.settings = settings
        if name == "haar":
            bundled = getattr(getattr(cv2, "data", None), "haarcascades", "")
            paths = [Path(bundled), Path(cv2.__file__).parent / "data",
                     Path("/usr/share/opencv4/haarcascades"), Path("/usr/share/opencv/haarcascades")]
            path = next((folder / "haarcascade_frontalface_default.xml" for folder in paths
                         if (folder / "haarcascade_frontalface_default.xml").is_file()), None)
            if not path:
                raise RuntimeError("Haar cascade missing: install opencv-data with apt")
            self.model = cv2.CascadeClassifier(str(path))
        elif name == "yunet":
            path = MODELS / "face_detection_yunet_2023mar.onnx"
            if not path.exists():
                raise RuntimeError("Download YuNet first: python download_models.py")
            self.model = cv2.FaceDetectorYN.create(str(path), "", (640, 480), settings["yunet_score"], 0.3, 5000)
        elif name == "dlib":
            import dlib
            self.model = dlib.get_frontal_face_detector()
        else:
            raise ValueError("Unsupported detector")

    def detect(self, image):
        if self.name == "haar":
            gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            return [{"box": list(map(int, box)), "score": None, "raw": None}
                    for box in self.model.detectMultiScale(gray, self.settings["haar_scale"],
                                                          self.settings["haar_neighbors"], minSize=(40, 40))]
        if self.name == "yunet":
            self.model.setInputSize((image.shape[1], image.shape[0]))
            _, faces = self.model.detect(image)
            return [] if faces is None else [
                {"box": [int(value) for value in row[:4]], "score": float(row[14]), "raw": row}
                for row in faces]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rectangles, scores, _ = self.model.run(gray, 0)
        return [{"box": [rect.left(), rect.top(), rect.width(), rect.height()],
                 "score": float(score), "raw": None} for rect, score in zip(rectangles, scores)]


@lru_cache(maxsize=1)
def dlib_health():
    try:
        import dlib
        dlib.get_frontal_face_detector()(np.zeros((64, 64), dtype=np.uint8), 0)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}. Rebuild dlib against this Python/NumPy; see README."


def capabilities():
    dlib_ok, dlib_error = dlib_health()
    return {"detectors": {"haar": True,
            "yunet": (MODELS / "face_detection_yunet_2023mar.onnx").exists(),
            "dlib": dlib_ok},
            "recognizers": {"lbph": hasattr(cv2, "face"),
            "sface": (MODELS / "face_recognition_sface_2021dec.onnx").exists(),
            "dlib": dlib_ok and all((MODELS / filename).exists()
                 for filename in ("shape_predictor_5_face_landmarks.dat", "dlib_face_recognition_resnet_model_v1.dat"))},
            "dlib_error": dlib_error}


class IdentityModel:
    def __init__(self, name, settings, detector, samples):
        self.name = name
        self.settings = settings
        self.labels = []
        self.features = []
        if name == "lbph":
            if not hasattr(cv2, "face"):
                raise RuntimeError("OpenCV contrib is required for LBPH")
            self.model = cv2.face.LBPHFaceRecognizer_create()
        elif name == "sface":
            self.model = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")
        elif name == "dlib":
            import dlib
            self.predictor = dlib.shape_predictor(str(MODELS / "shape_predictor_5_face_landmarks.dat"))
            self.model = dlib.face_recognition_model_v1(str(MODELS / "dlib_face_recognition_resnet_model_v1.dat"))
        else:
            raise ValueError("Unsupported recognizer")
        for sample in samples:
            image = decode(sample["image"])
            faces = detector.detect(image)
            if not faces:
                continue
            face = max(faces, key=lambda item: item["box"][2]*item["box"][3])
            self.features.append(self.feature(image, face))
            self.labels.append(sample["user_id"])
        if not self.labels:
            raise ValueError("No usable enrolled photos for this detector; add upright face images first")
        if name == "lbph":
            self.model.train(self.features, np.arange(len(self.features), dtype=np.int32))
        else:
            self.features = np.vstack(self.features)

    def feature(self, image, face):
        if self.name == "lbph":
            gray = cv2.cvtColor(crop(image, face["box"]), cv2.COLOR_BGR2GRAY)
            return cv2.equalizeHist(cv2.resize(gray, (100, 100)))
        if self.name == "sface":
            aligned = (self.model.alignCrop(image, face["raw"].reshape(1, -1)) if face["raw"] is not None
                       else cv2.resize(crop(image, face["box"]), (112, 112)))
            feature = self.model.feature(aligned).flatten()
            return feature / max(float(np.linalg.norm(feature)), 1e-10)
        import dlib
        left, top, width, height = face["box"]
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        shape = self.predictor(rgb, dlib.rectangle(left, top, left+width-1, top+height-1))
        return np.array(self.model.compute_face_descriptor(rgb, shape), dtype=np.float32)

    def identify(self, image, face):
        feature = self.feature(image, face)
        if self.name == "lbph":
            index, score = self.model.predict(feature)
            accepted = score <= self.settings["lbph_threshold"]
        elif self.name == "sface":
            scores = self.features @ feature
            index = int(np.argmax(scores))
            score = float(scores[index])
            accepted = score >= self.settings["sface_threshold"]
        else:
            distances = np.linalg.norm(self.features-feature, axis=1)
            index = int(np.argmin(distances))
            score = float(distances[index])
            accepted = score <= self.settings["dlib_threshold"]
        return self.labels[index] if accepted else None, float(score)


class Camera:
    def __init__(self, settings, source=None):
        self.capture = None
        self.picamera = None
        self.image = None
        self.first_frame = None
        self.selected = None
        self.rotation = settings["rotation"]
        if source and Path(source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            self.image = decode(Path(source).read_bytes())
        elif source:
            self.capture = cv2.VideoCapture(str(source))
            if not self.capture.isOpened():
                self.close()
                raise RuntimeError("Video cannot be opened")
        else:
            self.open_device(settings)

    @staticmethod
    def usb_indexes():
        if platform.system() != "Linux":
            return list(range(4))
        # Exclude Pi codec/ISP nodes; /dev/video0 alone does not imply a USB camera.
        return sorted(int(node.name[5:]) for node in Path("/sys/class/video4linux").glob("video*")
                      if node.name[5:].isdigit() and "/usb" in str(node.resolve()))

    def open_device(self, settings):
        mode, preferred = settings["camera"], settings["camera_index"]
        failures = []
        picamera_class = None
        if mode in ("auto", "pi"):
            try:
                from picamera2 import Picamera2
                picamera_class = Picamera2
                cameras = Picamera2.global_camera_info()
                indexes = ([preferred] if mode == "pi" else
                           [camera.get("Num", position) for position, camera in enumerate(cameras)
                            if "/usb" not in camera.get("Id", "")])
                indexes.sort(key=lambda index: index != preferred)
                for index in indexes:
                    try:
                        self.picamera = Picamera2(index)
                        self.picamera.configure(self.picamera.create_preview_configuration(
                            main={"size": (640, 480), "format": "RGB888"}))
                        self.picamera.start()
                        self.first_frame = self.pi_frame()
                        if self.first_frame is None or not self.first_frame.size:
                            raise RuntimeError("no frame received")
                        self.selected = {"kind": "pi", "index": index}
                        logging.getLogger(__name__).info("Camera connected: %s", self.selected)
                        return
                    except Exception as exc:
                        failures.append(f"Pi {index}: {exc}")
                        self.close()
            except Exception as exc:
                failures.append(f"Pi discovery: {exc}")
        if mode == "pi":
            if picamera_class is None:
                failures.append("Install python3-picamera2 with apt and use a --system-site-packages venv")
        else:
            indexes = [preferred] if mode == "usb" else self.usb_indexes()
            indexes.sort(key=lambda index: index != preferred)
            for index in indexes:
                try:
                    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
                    self.capture = cv2.VideoCapture(index, backend)
                    if not self.capture.isOpened():
                        raise RuntimeError("cannot open capture device")
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ok, self.first_frame = self.capture.read()
                    if not ok or self.first_frame is None or not self.first_frame.size:
                        raise RuntimeError("opened but returned no frame")
                    self.selected = {"kind": "usb", "index": index}
                    logging.getLogger(__name__).info("Camera connected: %s", self.selected)
                    return
                except Exception as exc:
                    failures.append(f"USB {index}: {exc}")
                    self.close()
        raise RuntimeError("No working camera found. " + ("; ".join(failures) or "No capture devices discovered.")
                           + " Stop other camera programs; run camera_diagnostic.py --no-preview for details.")

    def pi_frame(self):
        # A listed sensor can still fail to deliver frames. Do not wait forever.
        job = self.picamera.capture_array(wait=False)
        return self.picamera.wait(job, timeout=5)

    def read(self):
        if self.first_frame is not None:
            frame, self.first_frame = self.first_frame, None
        elif self.image is not None:
            frame, self.image = self.image, None
        elif self.picamera:
            frame = self.pi_frame()
        elif self.capture:
            ok, frame = self.capture.read()
            if not ok:
                return None
        else:
            return None
        if self.rotation:
            frame = cv2.rotate(frame, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotation])
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, round(frame.shape[0]*640/frame.shape[1])))
        return frame

    def close(self):
        picamera, self.picamera = self.picamera, None
        capture, self.capture = self.capture, None
        self.first_frame = None
        if picamera:
            try:
                picamera.stop()
            except Exception:
                logging.getLogger(__name__).warning("Camera stop failed", exc_info=True)
            finally:
                try:
                    picamera.close()
                except Exception:
                    logging.getLogger(__name__).warning("Camera close failed", exc_info=True)
        if capture:
            capture.release()
