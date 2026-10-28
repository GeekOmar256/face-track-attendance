import hmac
import os
import secrets
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, session
from werkzeug.exceptions import HTTPException

from cloud import CloudBridge
from engine import Engine
from exports import excel, report, statistics_csv
from store import Store
from vision import ROOT


def create_app(data_dir=None):
    data_dir = Path(data_dir or ROOT / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=30*1024*1024, SECRET_KEY=secrets.token_hex(32),
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")
    password = os.environ.get("FACETRACK_ADMIN_PASSWORD") or secrets.token_urlsafe(16)
    app.config["ADMIN_PASSWORD"] = password
    store = Store(data_dir / "attendance.sqlite3")
    engine = Engine(store)
    bridge = CloudBridge(engine)
    app.extensions.update(engine=engine, cloud=bridge, store=store)

    @app.before_request
    def authenticate():
        if request.path in ("/login", "/static/style.css"):
            return None
        if not session.get("admin"):
            if request.path == "/":
                return Response(Path(ROOT / "templates" / "login.html").read_text(), mimetype="text/html")
            return jsonify(error="Login required"), 401
        if request.method != "GET":
            supplied = request.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(supplied, session.get("csrf", "invalid")):
                return jsonify(error="Invalid CSRF token; reload the page"), 403

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, (ValueError, RuntimeError, KeyError, TypeError, ImportError)):
            return jsonify(error=str(exc)), 400
        app.logger.exception("Request failed")
        return jsonify(error="Internal error; check the server terminal"), 500

    attempts = {}

    @app.post("/login")
    def login():
        address = request.remote_addr
        recent = [stamp for stamp in attempts.get(address, []) if time.monotonic()-stamp < 60]
        attempts[address] = recent
        if len(recent) >= 10:
            return jsonify(error="Wait one minute before retrying"), 429
        data = request.get_json(silent=True) or {}
        if not hmac.compare_digest(str(data.get("password", "")), password):
            recent.append(time.monotonic())
            return jsonify(error="Incorrect password"), 401
        session.clear()
        session.update(admin=True, csrf=secrets.token_hex(24))
        return jsonify(ok=True)

    @app.post("/logout")
    def logout():
        session.clear()
        return jsonify(ok=True)

    @app.get("/")
    def index():
        return Response(Path(ROOT / "templates" / "index.html").read_text(), mimetype="text/html")

    @app.get("/api/state")
    def state():
        return jsonify(csrf=session["csrf"], users=store.users(), sessions=store.history(),
                       status=engine.status(), settings=engine.settings(),
                       cloud={"enabled": bridge.enabled, "error": bridge.error,
                              "pending": store.query("SELECT COUNT(*) AS count FROM outbox")[0]["count"]})

    @app.post("/api/settings")
    def settings():
        return jsonify(engine.save_settings(request.get_json()))

    @app.post("/api/users")
    def users():
        if request.is_json:
            body = request.get_json()
            return jsonify(engine.add_user(body, [], captured=True))
        images = [photo.read() for photo in request.files.getlist("photos")]
        if len(images) > 30:
            raise ValueError("Upload at most 30 images at once")
        return jsonify(engine.add_user(request.form, images))

    @app.delete("/api/users/<user_id>")
    def remove(user_id):
        engine.remove_user(user_id)
        return jsonify(ok=True)

    @app.patch("/api/users/<user_id>")
    def update_user(user_id):
        return jsonify(engine.update_user({**(request.get_json() or {}), "id": user_id}))

    @app.post("/api/device/<action>")
    def device_control(action):
        if action not in ("sleep", "wake"):
            raise ValueError("Unknown device action")
        return jsonify(getattr(engine, action)())

    @app.get("/api/users/<user_id>/photo")
    def photo(user_id):
        images = store.query("SELECT image FROM photos WHERE user_id=? LIMIT 1", (user_id,))
        if not images:
            return jsonify(error="No photo"), 404
        return Response(images[0]["image"], mimetype="image/jpeg")

    @app.post("/api/start")
    def start():
        if engine.running():
            raise ValueError("Stop the current session first")
        if request.is_json:
            body = request.get_json()
            if "source_path" in body:
                raise ValueError("Upload media through the file input")
        else:
            body = request.form.to_dict()
            media = request.files.get("source")
            if media and media.filename:
                suffix = Path(media.filename).suffix.lower()
                if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov", ".mkv"):
                    raise ValueError("Select an image or video")
                uploads = data_dir / "uploads"
                uploads.mkdir(exist_ok=True)
                path = uploads / (secrets.token_hex(12)+suffix)
                media.save(path)
                body["source_path"] = str(path)
        return jsonify(engine.start(body))

    @app.post("/api/stop")
    def stop():
        return jsonify(engine.stop())

    @app.get("/api/sessions/<session_id>")
    def detail(session_id):
        return jsonify(attendance=store.attendance(session_id), statistics=store.statistics(session_id),
                       events=store.recognized_people(session_id))

    @app.get("/api/sessions/<session_id>/frames")
    def frame_history(session_id):
        after = int(request.args.get("after", 0))
        return jsonify(store.query("SELECT * FROM frames WHERE session_id=? AND id>? ORDER BY id LIMIT 500",
                                   (session_id, after)))

    @app.get("/api/sessions/<session_id>/events")
    def event_history(session_id):
        after = int(request.args.get("after", 0))
        return jsonify(store.query("""SELECT id,frame_id,timestamp,user_id,user_name,class_name,detector,
                         recognizer,box,detector_score,match_score,accepted,recognition_ms
                         FROM events WHERE session_id=? AND id>? ORDER BY id LIMIT 500""", (session_id, after)))

    @app.get("/api/events/<int:event_id>/<kind>")
    def event_image(event_id, kind):
        if kind not in ("image", "thumbnail"):
            return jsonify(error="Not found"), 404
        rows = store.query(f"SELECT {kind} FROM events WHERE id=?", (event_id,))
        if not rows:
            return jsonify(error="Not found"), 404
        return Response(rows[0][kind], mimetype="image/jpeg")

    @app.get("/api/sessions/<session_id>/excel")
    def download_excel(session_id):
        return send_file(excel(store, session_id), download_name="attendance.xlsx", as_attachment=True)

    @app.get("/api/sessions/export-all.xlsx")
    def download_all_sessions():
        return send_file(excel(store), download_name="all-sessions.xlsx", as_attachment=True)

    @app.get("/api/sessions/<session_id>/report")
    def download_report(session_id):
        return Response(report(store, session_id), mimetype="text/html")

    @app.get("/api/sessions/<session_id>/statistics.csv")
    def download_stats(session_id):
        return Response(statistics_csv(store, session_id), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=statistics.csv"})

    @app.get("/frame.jpg")
    def frame():
        with engine.preview_lock:
            image = engine.preview
        if not image:
            return Response(status=204)
        return Response(image, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/camera/preview/start")
    def preview_start():
        return jsonify(engine.start_camera_preview())

    @app.post("/api/camera/preview/stop")
    def preview_stop():
        return jsonify(engine.stop_camera_preview())

    @app.get("/api/camera/preview/status")
    def preview_status():
        return jsonify(running=engine.camera_preview_running(), error=engine.camera_preview_error)

    @app.get("/api/camera/preview/stream")
    def preview_stream():
        def generate():
            while engine.camera_preview_running():
                with engine.preview_lock:
                    image = engine.camera_preview_jpeg
                if image:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(image)).encode()+b"\r\n\r\n"+image+b"\r\n")
                time.sleep(0.1)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    return app


if __name__ == "__main__":
    import argparse
    from waitress import serve
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app()
    try:
        app.extensions["cloud"].start()
    except Exception as exc:
        app.extensions["cloud"].error = str(exc)
        print(f"Firebase unavailable: {exc}; local attendance remains available")
    print(f"Local page: http://localhost:{args.port}")
    print(f"Admin password: {app.config['ADMIN_PASSWORD']}", flush=True)
    try:
        serve(app, host=args.host, port=args.port, threads=8)
    finally:
        app.extensions["engine"].stop()
        app.extensions["engine"].stop_camera_preview()
        app.extensions["cloud"].stop_signal.set()
