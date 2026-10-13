import base64
import json
import platform
import re
import threading
import time
import uuid

import cv2

from store import now
from vision import Camera, Detector, IdentityModel, capabilities, crop, decode, jpeg


DEFAULTS = {"detector": "haar", "recognizer": "lbph", "gate_enabled": True,
            "haar_scale": 1.1, "haar_neighbors": 5, "yunet_score": 0.7,
            "lbph_threshold": 70.0, "sface_threshold": 0.363, "dlib_threshold": 0.6,
            "camera": "auto", "camera_index": 0, "rotation": 0, "interval_ms": 100,
            "mode": "live"}


class Engine:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.stop_signal = threading.Event()
        self.thread = None
        self.session_id = None
        self.preview = None
        self.preview_lock = threading.Lock()
        self.error = None
        self.initializing = False
        self.camera_preview_thread = None
        self.camera_preview_stop = threading.Event()
        self.camera_preview_frame = None
        self.camera_preview_jpeg = None
        self.camera_preview_error = None

    def settings(self):
        rows = self.store.query("SELECT value FROM settings WHERE id=1")
        return {**DEFAULTS, **(json.loads(rows[0]["value"]) if rows else {})}

    def save_settings(self, data):
        with self.lock:
            if self.running() or self.camera_preview_running():
                raise ValueError("Stop the session before changing settings")
            settings = self.validate({**self.settings(), **data})
            self.store.execute("INSERT OR REPLACE INTO settings VALUES (1,?)", (json.dumps(settings),))
            self.store.enqueue("settings", "current", settings)
            return settings

    @staticmethod
    def validate(settings):
        if settings["detector"] not in ("haar", "yunet", "dlib") or settings["recognizer"] not in ("lbph", "sface", "dlib"):
            raise ValueError("Unknown algorithm")
        if settings["mode"] not in ("live", "single") or settings["camera"] not in ("auto", "pi", "usb"):
            raise ValueError("Invalid camera or mode")
        for key, low, high in (("haar_scale", 1.01, 2), ("haar_neighbors", 1, 30),
                              ("yunet_score", 0.01, 1), ("lbph_threshold", 0, 1000),
                              ("sface_threshold", -1, 1), ("dlib_threshold", 0.01, 2),
                              ("interval_ms", 0, 10000), ("camera_index", 0, 20)):
            value = float(settings[key])
            if not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            settings[key] = int(value) if key in ("haar_neighbors", "interval_ms", "camera_index") else value
        settings["rotation"] = int(settings["rotation"])
        if settings["rotation"] not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180, or 270")
        if not isinstance(settings["gate_enabled"], bool):
            raise ValueError("gate_enabled must be boolean")
        return {key: settings[key] for key in DEFAULTS}

    def running(self):
        return self.initializing or bool(self.thread and self.thread.is_alive())

    def add_user(self, details, images, captured=False):
        with self.lock:
            if self.running():
                raise ValueError("Stop attendance before enrolling a person")
            user_id = str(details.get("id", "")).strip()
            name = str(details.get("name", "")).strip()
            class_name = str(details.get("class_name", "")).strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", user_id) or not name or not class_name:
                raise ValueError("ID (letters/numbers/-/_), name, and class are required")
            if len(name) > 120 or len(class_name) > 120:
                raise ValueError("Name or class is too long")
            if not images and not captured:
                raise ValueError("Select at least one photograph")
            settings = self.settings()
            detector = Detector(settings["detector"], settings)
            if captured:
                if self.camera_preview_running():
                    with self.preview_lock:
                        frame = self.camera_preview_frame.copy() if self.camera_preview_frame is not None else None
                    if frame is None:
                        raise ValueError("Wait for a live preview frame before capturing")
                    images = [jpeg(frame)]
                else:
                    camera = Camera(settings)
                    try:
                        frame = camera.read()
                        if frame is None:
                            raise ValueError("Camera returned no frame")
                        images = [jpeg(frame)]
                    finally:
                        camera.close()
            accepted = []
            for content in images:
                if len(content) > 10*1024*1024:
                    raise ValueError("Each photo must be smaller than 10 MB")
                frame = decode(content)
                faces = detector.detect(frame)
                if len(faces) != 1:
                    ordered = sorted(faces, key=lambda item: item["box"][2]*item["box"][3], reverse=True)
                    if len(ordered) < 2 or ordered[0]["box"][2]*ordered[0]["box"][3] < 2*ordered[1]["box"][2]*ordered[1]["box"][3]:
                        raise ValueError(f"Enrollment photo contains {len(faces)} detections; use a clear single-person photo or another detector")
                accepted.append(jpeg(frame))
            with self.store.connect() as db:
                db.execute("""INSERT INTO users VALUES (?,?,?,1,?) ON CONFLICT(id)
                            DO UPDATE SET name=excluded.name,class_name=excluded.class_name,active=1""",
                           (user_id, name, class_name, now()))
                for content in accepted:
                    db.execute("INSERT INTO photos(user_id,image,created) VALUES (?,?,?)", (user_id, content, now()))
                self.store.enqueue("users", user_id, {"id": user_id, "name": name, "class_name": class_name,
                                   "active": True, "thumbnail": base64.b64encode(jpeg(decode(accepted[0]), 160)).decode()}, db)
            return {"id": user_id, "added_photos": len(accepted)}

    def remove_user(self, user_id):
        with self.lock:
            if self.running():
                raise ValueError("Stop attendance before removing a person")
            with self.store.connect() as db:
                exists = db.execute("SELECT id FROM users WHERE id=? AND active=1", (user_id,)).fetchone()
                if not exists:
                    raise ValueError("User does not exist")
                db.execute("UPDATE users SET active=0 WHERE id=?", (user_id,))
                db.execute("DELETE FROM photos WHERE user_id=?", (user_id,))
                self.store.enqueue("users", user_id, {"id": user_id, "active": False}, db)

    def start(self, request):
        with self.lock:
            if self.running():
                raise ValueError("A session is already running")
            self.stop_camera_preview()
            name = str(request.get("name", "Attendance")).strip()[:120]
            class_name = str(request.get("class_name", "")).strip()[:120]
            settings = self.validate({**self.settings(), "mode": request.get("mode", self.settings()["mode"])})
            available = capabilities()
            if not available["detectors"][settings["detector"]] or not available["recognizers"][settings["recognizer"]]:
                raise ValueError("Selected model unavailable; run doctor.py and download_models.py")
            people = {person["id"]: person for person in self.store.users()
                      if not class_name or person["class_name"] == class_name}
            if not people:
                raise ValueError("No enrolled users in this class")
            samples = self.store.query("SELECT p.user_id,p.image FROM photos p JOIN users u ON u.id=p.user_id WHERE u.active=1")
            samples = [sample for sample in samples if sample["user_id"] in people]
            source = request.get("source_path")
            if source:
                settings["source"] = "uploaded file"
            else:
                settings["source"] = settings["camera"]
            settings["opencv"] = cv2.__version__
            settings["platform"] = platform.platform()
            settings["python"] = platform.python_version()
            session_id = uuid.uuid4().hex
            self.session_id = session_id
            self.initializing = True
            self.error = None
            with self.preview_lock:
                self.preview = None
            self.stop_signal.clear()
            timestamp = now()
            self.store.execute("INSERT INTO sessions VALUES (?,?,?,?,NULL,'running',NULL,?)",
                               (session_id, name, class_name, timestamp, json.dumps(settings)))
            self.store.enqueue("sessions", session_id, {"id": session_id, "name": name,
                               "class_name": class_name, "started": timestamp, "state": "running", "config": settings})
            self.thread = threading.Thread(target=self._run, args=(session_id, settings, source, samples, people), daemon=True)
            self.thread.start()
            return {"session_id": session_id}

    def stop(self):
        self.stop_signal.set()
        if self.thread:
            self.thread.join(timeout=3)
        return {"stopping": self.running()}

    def camera_preview_running(self):
        return bool(self.camera_preview_thread and self.camera_preview_thread.is_alive())

    def start_camera_preview(self):
        with self.lock:
            if self.running():
                raise ValueError("Stop attendance before opening the enrollment camera")
            if self.camera_preview_running():
                return {"running": True}
            self.camera_preview_stop.clear()
            self.camera_preview_error = None
            with self.preview_lock:
                self.camera_preview_frame = None
                self.camera_preview_jpeg = None
            settings = self.settings()
            self.camera_preview_thread = threading.Thread(target=self._preview_camera, args=(settings,), daemon=True)
            self.camera_preview_thread.start()
            return {"running": True}

    def stop_camera_preview(self):
        self.camera_preview_stop.set()
        if self.camera_preview_thread:
            self.camera_preview_thread.join(timeout=3)
        if self.camera_preview_running():
            raise ValueError("Camera is still stopping; wait before starting another operation")
        return {"running": False}

    def _preview_camera(self, settings):
        camera = None
        try:
            camera = Camera(settings)
            while not self.camera_preview_stop.is_set():
                frame = camera.read()
                if frame is None:
                    raise ValueError("Camera returned no preview frames")
                encoded = jpeg(frame, 640)
                with self.preview_lock:
                    self.camera_preview_frame = frame.copy()
                    self.camera_preview_jpeg = encoded
                self.camera_preview_stop.wait(0.08)
        except Exception as exc:
            self.camera_preview_error = str(exc)
        finally:
            if camera:
                try:
                    camera.close()
                except Exception as exc:
                    self.camera_preview_error = str(exc)

    def _run(self, session_id, settings, source, samples, people):
        camera = None
        state = "finished"
        try:
            detector = Detector(settings["detector"], settings)
            gate = Detector("haar", settings) if settings["gate_enabled"] else None
            identity = IdentityModel(settings["recognizer"], settings, detector, samples)
            camera = Camera(settings, source)
            self.initializing = False
            frame_number = 0
            while not self.stop_signal.is_set():
                cycle = time.perf_counter()
                frame = camera.read()
                if frame is None:
                    break
                frame_number += 1
                self.process_frame(session_id, frame_number, frame, settings, detector, identity, people, gate)
                if settings["mode"] == "single":
                    break
                self.stop_signal.wait(max(0, settings["interval_ms"]/1000-(time.perf_counter()-cycle)))
        except Exception as exc:
            state = "error"
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if camera:
                try:
                    camera.close()
                except Exception as exc:
                    state = "error"
                    self.error = str(exc)
            self.initializing = False
            ended = now()
            self.store.execute("UPDATE sessions SET ended=?,state=?,error=? WHERE id=?", (ended, state, self.error, session_id))
            self.store.enqueue("sessions", session_id, {"state": state, "ended": ended, "error": self.error,
                               "statistics": self.store.statistics(session_id)})

    def process_frame(self, session_id, frame_number, frame, settings, detector, identity, people, gate=None):
        started = time.perf_counter()
        timestamp = now()
        gate_faces = gate.detect(frame) if gate else []
        gate_ms = (time.perf_counter()-started)*1000
        detect_start = time.perf_counter()
        faces = detector.detect(frame) if not gate or gate_faces else []
        detect_ms = (time.perf_counter()-detect_start)*1000
        annotated = frame.copy()
        matches = []
        recognition_ms = 0
        for face in faces:
            match_start = time.perf_counter()
            user_id, score = identity.identify(frame, face)
            elapsed = (time.perf_counter()-match_start)*1000
            recognition_ms += elapsed
            person = people.get(user_id)
            left, top, width, height = face["box"]
            cv2.rectangle(annotated, (left, top), (left+width, top+height), (40, 180, 50) if person else (30, 90, 220), 2)
            cv2.putText(annotated, person["name"] if person else "Unknown", (max(left, 0), max(top-8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 180, 50), 2)
            matches.append((face, person, score, elapsed, jpeg(crop(frame, face["box"]), 120)))
        image = jpeg(annotated, 640)
        pipeline_ms = (time.perf_counter()-started)*1000
        fps = 1000 / max(pipeline_ms, 0.001)
        write_start = time.perf_counter()
        with self.store.connect() as db:
            frame_id = db.execute("""INSERT INTO frames(session_id,timestamp,frame_number,gate_ms,detect_ms,
                recognition_ms,pipeline_ms,fps,width,height,gate_faces,faces,config)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (session_id, timestamp, frame_number, gate_ms, detect_ms,
                recognition_ms, pipeline_ms, fps, frame.shape[1], frame.shape[0], len(gate_faces), len(faces), json.dumps(settings))).lastrowid
            for face, person, score, elapsed, thumbnail in matches:
                event = {"session_id": session_id, "timestamp": timestamp, "user_id": person["id"] if person else None,
                         "user_name": person["name"] if person else "Unknown", "class_name": person["class_name"] if person else None,
                         "detector": settings["detector"], "recognizer": settings["recognizer"], "box": face["box"],
                         "detector_score": face["score"], "match_score": score, "accepted": bool(person),
                         "recognition_ms": elapsed, "frame_id": frame_id,
                         "thumbnail": base64.b64encode(thumbnail).decode()}
                event_id = db.execute("""INSERT INTO events(frame_id,session_id,timestamp,user_id,user_name,class_name,
                    detector,recognizer,box,detector_score,match_score,accepted,recognition_ms,thumbnail,image)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (frame_id, session_id, timestamp, event["user_id"], event["user_name"],
                    event["class_name"], settings["detector"], settings["recognizer"], json.dumps(face["box"]), face["score"], score,
                    int(bool(person)), elapsed, thumbnail, image)).lastrowid
                event["id"] = event_id
                self.store.enqueue("events", event_id, event, db)
                if person:
                    result = db.execute("""INSERT OR IGNORE INTO attendance(session_id,user_id,name,class_name,timestamp,event_id)
                                        VALUES (?,?,?,?,?,?)""", (session_id, person["id"], person["name"], person["class_name"], timestamp, event_id))
                    if result.rowcount:
                        attendance = {**event, "id": f"{session_id}_{person['id']}", "name": person["name"],
                                      "image": base64.b64encode(jpeg(annotated, 320)).decode()}
                        self.store.enqueue("attendance", attendance["id"], attendance, db)
        write_ms = (time.perf_counter()-write_start)*1000
        self.store.execute("UPDATE frames SET write_ms=? WHERE id=?", (write_ms, frame_id))
        self.store.enqueue("frames", frame_id, {"id": frame_id, "session_id": session_id, "timestamp": timestamp,
                          "frame_number": frame_number, "gate_ms": gate_ms, "detect_ms": detect_ms,
                          "recognition_ms": recognition_ms, "pipeline_ms": pipeline_ms, "fps": fps,
                          "write_ms": write_ms, "faces": len(faces), "gate_faces": len(gate_faces),
                          "width": frame.shape[1], "height": frame.shape[0], "config": settings})
        with self.preview_lock:
            self.preview = image
        return {"frame_id": frame_id, "faces": len(faces), "fps": fps}

    def status(self):
        return {"running": self.running(), "initializing": self.initializing, "session_id": self.session_id,
                "error": self.error, "statistics": self.store.statistics(self.session_id) if self.session_id else {},
                "capabilities": capabilities(), "camera_preview_running": self.camera_preview_running()}
