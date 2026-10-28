import base64
import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone

from store import now


class CloudBridge:
    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store
        self.enabled = False
        self.error = None
        self.stop_signal = threading.Event()
        self.thread = None

    def start(self):
        config_path = Path(__file__).parent / "data" / "firebase-settings.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or config.get("credentials")
        device_id = os.environ.get("FACETRACK_DEVICE_ID") or config.get("device_id")
        if not credentials or not device_id:
            return
        import firebase_admin
        from firebase_admin import credentials as firebase_credentials, firestore
        app = firebase_admin.initialize_app(firebase_credentials.Certificate(credentials))
        self.client = firestore.client(app=app)
        self.device = self.client.collection("face_track_devices").document(device_id)
        self.store.enqueue("settings", "current", self.engine.settings())
        for user in self.store.users():
            image = self.store.query("SELECT image FROM photos WHERE user_id=? LIMIT 1", (user["id"],))
            payload = {**user, "active": True}
            if image:
                from vision import decode, jpeg
                payload["thumbnail"] = base64.b64encode(jpeg(decode(image[0]["image"]), 160)).decode()
            self.store.enqueue("users", user["id"], payload)
        self.enabled = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_signal.is_set():
            try:
                self.device.set({"status": self.engine.status(), "last_seen": now()}, merge=True)
                pending = self.device.collection("commands").where("state", "==", "pending").limit(10).stream()
                for document in pending:
                    self._command(document)
                items = self.store.query("SELECT * FROM outbox ORDER BY id LIMIT 100")
                if items:
                    batch = self.client.batch()
                    for item in items:
                        reference = self.device.collection(item["collection"]).document(item["document"])
                        batch.set(reference, json.loads(item["payload"]), merge=True)
                    batch.commit()
                    with self.store.connect() as db:
                        db.executemany("DELETE FROM outbox WHERE id=?", [(item["id"],) for item in items])
                self.error = None
            except Exception as exc:
                self.error = str(exc)
            self.stop_signal.wait(2)

    def _command(self, document):
        prior = self.store.query("SELECT result FROM commands WHERE id=?", (document.id,))
        if prior:
            document.reference.update(json.loads(prior[0]["result"]))
            return
        body = document.to_dict()
        action = body.get("action")
        payload = body.get("payload", {})
        result = {"state": "done", "completed": now()}
        self.store.execute("INSERT INTO commands VALUES (?,?)", (document.id, json.dumps(
            {"state": "error", "error": "Execution interrupted; check device state before resubmitting"})))
        try:
            created = body.get("created")
            if not isinstance(created, datetime) or abs((datetime.now(timezone.utc)-created).total_seconds()) > 300:
                raise ValueError("Command expired; send a new command")
            if action == "start":
                if "source_path" in payload:
                    raise ValueError("Remote sessions use the device camera")
                result["result"] = self.engine.start(payload)
            elif action == "stop":
                result["result"] = self.engine.stop()
            elif action == "sleep":
                result["result"] = self.engine.sleep()
            elif action == "wake":
                result["result"] = self.engine.wake()
            elif action == "update_user":
                result["result"] = self.engine.update_user(payload)
            elif action == "settings":
                result["result"] = self.engine.save_settings(payload)
            elif action == "remove_user":
                self.engine.remove_user(str(payload["id"]))
            elif action == "add_user":
                photos = payload.get("photos", [])
                if len(photos) > 5 or sum(len(photo) for photo in photos) > 700000:
                    raise ValueError("Send up to five resized photos totaling less than 525 KB")
                images = [base64.b64decode(photo, validate=True) for photo in photos]
                result["result"] = self.engine.add_user(payload, images, bool(payload.get("capture")))
            elif action in ("approve_account", "reject_account"):
                result["result"] = self._account_access(str(payload.get("uid", "")), action == "approve_account")
            else:
                raise ValueError("Unknown command")
        except Exception as exc:
            result.update(state="error", error=str(exc))
        self.store.execute("INSERT OR REPLACE INTO commands VALUES (?,?)", (document.id, json.dumps(result)))
        document.reference.update(result)

    def _account_access(self, uid, approve):
        from firebase_admin import auth
        if not uid or "/" in uid or len(uid) > 128:
            raise ValueError("Invalid account ID")
        reference = self.client.collection("face_track_accounts").document(uid)
        snapshot = reference.get()
        profile = snapshot.to_dict() if snapshot.exists else None
        if not profile or profile.get("device_id") != self.device.id or profile.get("state") != "pending":
            raise ValueError("No pending account request for this device")
        if approve:
            user = auth.get_user(uid)
            if user.email != profile.get("email") or user.disabled:
                raise ValueError("Account details do not match the access request")
            auth.set_custom_user_claims(uid, {**(user.custom_claims or {}), "face_track_admin": True,
                                            "face_track_device": self.device.id})
        state = "approved" if approve else "rejected"
        reference.update({"state": state, "reviewed": now()})
        return {"uid": uid, "state": state}
