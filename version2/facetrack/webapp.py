"""Browser front end for the Face Track pipeline.

Built on Python's own http.server, so it adds no dependency at all: the
Raspberry Pi still needs nothing beyond opencv-contrib-python and numpy. The
page is served to any browser on the same Wi-Fi, which matches the application
layer described in the FYP1 plan, where the dashboard reaches the Raspberry Pi
over the local network.

It also means no display is needed on the Pi itself. A headless board with no
desktop installed can serve this page, and the video arrives as JPEG frames
over HTTP rather than as X11 traffic forwarded through SSH.
"""

from __future__ import annotations

import csv
import io
import json
import socket

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from . import config
from .detectors import DETECTOR_NAMES
from .recognizers import RECOGNIZER_NAMES, default_model_path
from .session import PipelineSession

BOUNDARY = "facetrackframe"
UPLOAD_DIR = config.DATA_DIR / "uploads"
MAX_UPLOAD = 512 * 1024 * 1024          # 512 MB, enough for a classroom video


# ======================================================================= page
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Track</title>
<style>
  :root {
    --bg:#11141a; --panel:#1a1f28; --line:#2b323e; --text:#e8ecf2;
    --muted:#98a2b3; --accent:#3da9fc; --ok:#35c26a; --bad:#ef4b4b; --warn:#f0a132;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 16px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:17px; letter-spacing:.2px; }
  header span { color:var(--muted); font-size:12px; }
  .wrap { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:16px; padding:16px; }
  .wrap > * { min-width:0; }
  img { max-width:100%; }
  @media (max-width:900px) { .wrap { grid-template-columns:1fr; padding:12px; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:14px; min-width:0; overflow:hidden; }
  .card h2 { margin:0 0 10px; font-size:12px; text-transform:uppercase;
             letter-spacing:.8px; color:var(--muted); font-weight:600; }
  .viewbox { position:relative; background:#000; border-radius:8px; overflow:hidden;
             min-height:360px; max-height:70vh; display:flex;
             align-items:center; justify-content:center; }
  /* max-width/max-height rather than 100% plus object-fit: this letterboxes a
     tall portrait frame reliably, whatever the source aspect ratio is.
     min-width:0 is needed because a flex item otherwise refuses to shrink below
     its natural width, which overflows the card on a phone. */
  #view { max-width:100%; max-height:70vh; width:auto; height:auto;
          display:block; min-width:0; }
  #view:not([src]) { visibility:hidden; }
  #placeholder { position:absolute; color:var(--muted); font-size:13px; }
  label { display:block; margin:8px 0 3px; font-size:12px; color:var(--muted); }
  select, input[type=text], input[type=number], input[type=file] {
    width:100%; padding:7px 8px; background:#131820; color:var(--text);
    border:1px solid var(--line); border-radius:6px; font-size:13px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; } .row > * { flex:1 1 120px; min-width:0; }
  .radios { display:flex; gap:6px; flex-wrap:wrap; margin-top:4px; }
  .radios label { display:flex; align-items:center; gap:5px; margin:0;
                  padding:6px 9px; background:#131820; border:1px solid var(--line);
                  border-radius:6px; cursor:pointer; color:var(--text); font-size:13px; }
  .radios input { accent-color:var(--accent); }
  button { width:100%; padding:9px; margin-top:10px; border:0; border-radius:7px;
           background:var(--accent); color:#04121f; font-weight:700; cursor:pointer;
           font-size:13px; }
  button.ghost { background:#222a36; color:var(--text); font-weight:600; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
  @media (max-width:620px) { .grid { grid-template-columns:repeat(2,1fr); } }
  .stat { background:#131820; border:1px solid var(--line); border-radius:8px; padding:9px 10px; }
  .stat .k { font-size:11px; color:var(--muted); }
  .stat .v { font-size:18px; font-weight:700; margin-top:2px; }
  table { width:100%; border-collapse:collapse; margin-top:6px; font-size:13px; }
  th { text-align:left; color:var(--muted); font-weight:600; font-size:11px;
       text-transform:uppercase; letter-spacing:.5px; padding:4px 6px; }
  td { padding:5px 6px; border-top:1px solid var(--line); }
  .pill { padding:2px 8px; border-radius:99px; font-size:11px; font-weight:700; }
  .ok { background:rgba(53,194,106,.15); color:var(--ok); }
  .bad { background:rgba(239,75,75,.15); color:var(--bad); }
  .muted { color:var(--muted); }
  #msg { margin-top:10px; font-size:12px; min-height:1.4em; }
  .hide { display:none !important; }
</style>
</head>
<body>
<header>
  <h1>Face Track</h1>
  <span>detection and recognition &middot; FYP2</span>
  <span id="state" style="margin-left:auto"></span>
</header>

<div class="wrap">
  <div>
    <div class="card">
      <h2>Preview</h2>
      <div class="viewbox">
        <div id="placeholder">not running</div>
        <img id="view" alt="">
      </div>
      <div class="row" style="margin-top:8px">
        <button class="ghost" id="snap">Save snapshot</button>
        <button class="ghost" id="reset">Reset statistics</button>
        <button class="ghost" id="csv">Export CSV</button>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <h2>Statistics</h2>
      <div class="grid" id="stats"></div>
      <table>
        <thead><tr><th>Identified</th><th>ID</th><th>Frames</th><th>Share</th></tr></thead>
        <tbody id="people"><tr><td colspan="4" class="muted">nothing yet</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Source</h2>
    <div class="radios" id="srcKind">
      <label><input type="radio" name="src" value="camera" checked> Camera</label>
      <label><input type="radio" name="src" value="video"> Video</label>
      <label><input type="radio" name="src" value="photos"> Photos</label>
    </div>

    <div id="cameraBox">
      <label>Camera index</label>
      <input type="number" id="cameraIndex" value="0" min="0" max="8">
    </div>

    <div id="fileBox" class="hide">
      <label>Upload a file to the server</label>
      <input type="file" id="upload">
      <label>Or a path already on the server</label>
      <select id="serverFile"></select>
      <label style="display:flex;align-items:center;gap:6px;margin-top:8px">
        <input type="checkbox" id="loop" style="width:auto"> Loop when it ends
      </label>
    </div>

    <h2 style="margin-top:16px">Detection</h2>
    <div class="radios" id="detKind">
      <label><input type="radio" name="det" value="haar" checked> Haar</label>
      <label><input type="radio" name="det" value="yunet"> YuNet</label>
    </div>
    <div id="haarOpts">
      <div class="row">
        <div><label>scaleFactor</label><input type="number" id="scaleFactor" value="1.1" step="0.01" min="1.01"></div>
        <div><label>minNeighbors</label><input type="number" id="minNeighbors" value="5" min="1"></div>
      </div>
      <label>Minimum face size (px)</label>
      <input type="number" id="minSize" value="60" min="20" step="10">
    </div>
    <div id="yunetOpts" class="hide">
      <label>Score threshold</label>
      <input type="number" id="scoreThreshold" value="0.9" step="0.05" min="0.05" max="1">
    </div>

    <h2 style="margin-top:16px">Recognition</h2>
    <select id="recognizer">
      <option value="none">Off, detection only</option>
    </select>
    <label>Threshold <span class="muted">(blank uses config.py)</span></label>
    <input type="text" id="threshold" placeholder="default">

    <button id="start">Start</button>
    <button id="stop" class="ghost" disabled>Stop</button>
    <div id="msg" class="muted"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const msg = (t, bad) => { $("msg").textContent = t || ""; $("msg").style.color = bad ? "var(--bad)" : "var(--muted)"; };
let running = false;

function sourceKind() { return document.querySelector('input[name=src]:checked').value; }
function detKind()    { return document.querySelector('input[name=det]:checked').value; }

$("srcKind").onchange = () => {
  const camera = sourceKind() === "camera";
  $("cameraBox").classList.toggle("hide", !camera);
  $("fileBox").classList.toggle("hide", camera);
  loadFiles();
};
$("detKind").onchange = () => {
  const haar = detKind() === "haar";
  $("haarOpts").classList.toggle("hide", !haar);
  $("yunetOpts").classList.toggle("hide", haar);
};

async function loadConfig() {
  const cfg = await (await fetch("api/config")).json();
  const sel = $("recognizer");
  sel.innerHTML = '<option value="none">Off, detection only</option>';
  for (const r of cfg.recognizers) {
    const o = document.createElement("option");
    o.value = r.name;
    o.textContent = r.label + (r.trained ? "" : "  (not trained yet)");
    o.disabled = !r.trained;
    sel.appendChild(o);
  }
  if (!cfg.yunet_available) {
    document.querySelector('input[name=det][value=yunet]').disabled = true;
  }
  loadFiles();
}

async function loadFiles() {
  const files = await (await fetch("api/files")).json();
  const sel = $("serverFile");
  const kind = sourceKind();
  const list = kind === "video" ? files.videos : files.photos;
  sel.innerHTML = '<option value="">-- choose --</option>';
  for (const f of list) {
    const o = document.createElement("option");
    o.value = f.path; o.textContent = f.label;
    sel.appendChild(o);
  }
}

$("upload").onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  msg("uploading " + file.name + " ...");
  const res = await fetch("api/upload?name=" + encodeURIComponent(file.name),
                          { method: "POST", body: file });
  const out = await res.json();
  if (!res.ok) { msg(out.error || "upload failed", true); return; }
  msg("uploaded " + out.name);
  await loadFiles();
  $("serverFile").value = out.path;
};

$("start").onclick = async () => {
  const body = {
    detector: detKind(),
    recognizer: $("recognizer").value,
    threshold: $("threshold").value.trim() || null,
    scale_factor: parseFloat($("scaleFactor").value),
    min_neighbors: parseInt($("minNeighbors").value),
    min_size: parseInt($("minSize").value),
    score_threshold: parseFloat($("scoreThreshold").value),
    loop: $("loop").checked
  };
  if (sourceKind() === "camera") {
    body.camera_index = parseInt($("cameraIndex").value);
  } else {
    body.source = $("serverFile").value;
    if (!body.source) { msg("choose or upload a file first", true); return; }
  }
  msg("starting ...");
  const res = await fetch("api/start", { method:"POST", body: JSON.stringify(body) });
  const out = await res.json();
  if (!res.ok) { msg(out.error || "could not start", true); return; }
  msg(out.detector_description || "running");
  attachStream();
};

function attachStream() {
  $("view").src = "stream.mjpg?t=" + Date.now();
  $("placeholder").textContent = "";
  running = true; $("start").disabled = true; $("stop").disabled = false;
}

function detachStream(label) {
  $("view").removeAttribute("src");
  $("placeholder").textContent = label;
  running = false; $("start").disabled = false; $("stop").disabled = true;
}

$("stop").onclick = async () => {
  await fetch("api/stop", { method:"POST" });
  detachStream("stopped");
  msg("stopped, statistics kept");
};

$("reset").onclick = async () => { await fetch("api/reset", {method:"POST"}); msg("statistics reset"); };
$("snap").onclick  = async () => {
  const out = await (await fetch("api/snapshot", {method:"POST"})).json();
  msg(out.path ? "saved " + out.path : (out.error || "nothing to save"), !out.path);
};
$("csv").onclick = () => { window.location = "api/export.csv"; };

const FIELDS = [
  ["frames","Frames"], ["total_faces","Faces"], ["detection_rate","Detection rate"],
  ["throughput_fps","Throughput"], ["mean_detect_ms","Detection"],
  ["mean_pipeline_ms","Pipeline mean"], ["max_pipeline_ms","Pipeline max"],
  ["meets_nfreq3","NFReq-3"]
];
const UNITS = { detection_rate:"%", throughput_fps:" fps", mean_detect_ms:" ms",
                mean_pipeline_ms:" ms", max_pipeline_ms:" ms" };

async function poll() {
  try {
    const s = await (await fetch("api/status")).json();
    $("state").innerHTML = s.state === "running"
      ? '<span class="pill ok">running</span>'
      : '<span class="pill ' + (s.state === "error" ? "bad" : "") + '">' + s.state + '</span>';
    // Reattach after a page reload, so refreshing during a run keeps the preview.
    if (s.state === "running" && !running) attachStream();
    if (s.state !== "running" && running) {
      detachStream(s.state === "error" ? "error" : "finished");
      if (s.state === "error") msg(s.error, true);
      else if (s.state === "finished") msg("source finished");
    }
    const st = s.stats;
    $("stats").innerHTML = FIELDS.map(([k, label]) => {
      let v = st[k];
      if (k === "meets_nfreq3")
        v = st.frames ? (v ? '<span class="pill ok">PASS</span>' : '<span class="pill bad">FAIL</span>') : "-";
      else v = v + (UNITS[k] || "");
      return '<div class="stat"><div class="k">' + label + '</div><div class="v">' + v + '</div></div>';
    }).join("");

    const rows = st.people.map(p =>
      "<tr><td>" + p.name + "</td><td class='muted'>" + (p.id || "-") +
      "</td><td>" + p.frames + "</td><td class='muted'>" + p.share + "%</td></tr>").join("");
    const unknown = st.unknown
      ? "<tr><td class='muted'>Unknown</td><td class='muted'>-</td><td>" + st.unknown + "</td><td class='muted'>-</td></tr>"
      : "";
    $("people").innerHTML = (rows + unknown) || "<tr><td colspan='4' class='muted'>nothing yet</td></tr>";
  } catch (e) { /* server busy, try again next tick */ }
  setTimeout(poll, 700);
}

loadConfig(); poll();
</script>
</body>
</html>
"""


# ==================================================================== handler
class Handler(BaseHTTPRequestHandler):
    server_version = "FaceTrack/0.1"
    session: PipelineSession = None          # set by serve()

    def log_message(self, fmt, *args):       # keep the console readable
        if self.path.startswith(("/stream", "/api/status")):
            return
        super().log_message(fmt, *args)

    # ------------------------------------------------------------- helpers
    def _json(self, payload, status=200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # ------------------------------------------------------------- routing
    def do_GET(self) -> None:
        route = urlparse(self.path)
        path = route.path.rstrip("/") or "/"
        try:
            if path == "/":
                self._send_page()
            elif path == "/stream.mjpg":
                self._send_stream()
            elif path == "/api/status":
                self._json(self.session.status())
            elif path == "/api/config":
                self._json(self._config_payload())
            elif path == "/api/files":
                self._json(self._files_payload())
            elif path == "/api/export.csv":
                self._send_csv()
            else:
                self._json({"error": "not found"}, 404)
        except ConnectionError:
            pass                                  # the browser navigated away

    def do_POST(self) -> None:
        route = urlparse(self.path)
        path = route.path.rstrip("/") or "/"
        try:
            if path == "/api/start":
                self._start(self._read_json())
            elif path == "/api/stop":
                self.session.stop()
                self._json({"ok": True})
            elif path == "/api/reset":
                self.session.stats.reset()
                self._json({"ok": True})
            elif path == "/api/snapshot":
                saved = self.session.save_snapshot()
                self._json({"path": str(saved)} if saved
                           else {"error": "nothing to save yet"})
            elif path == "/api/upload":
                self._upload(parse_qs(route.query))
            else:
                self._json({"error": "not found"}, 404)
        except ConnectionError:
            pass

    # --------------------------------------------------------------- pages
    def _send_page(self) -> None:
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()

        last_id = -1
        idle = 0
        while True:
            if self.session.state not in ("running",):
                idle += 1
                if idle > 40:                       # about four seconds
                    break
            current = self.session.frame_id
            if current == last_id:
                time.sleep(0.02)
                continue
            last_id = current
            idle = 0
            jpeg = self.session.latest_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            self.wfile.write(f"--{BOUNDARY}\r\n".encode())
            self.wfile.write(b"Content-Type: image/jpeg\r\n")
            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
            self.wfile.write(jpeg)
            self.wfile.write(b"\r\n")

    def _send_csv(self) -> None:
        status = self.session.status()
        rows = self.session.stats.export_rows(
            status["detector"], status["recognizer"], status["source"])
        buffer = io.StringIO()
        if rows:
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        body = buffer.getvalue().encode("utf-8")
        name = f"facetrack_session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------- payloads
    @staticmethod
    def _config_payload() -> dict:
        recognizers = []
        for name in RECOGNIZER_NAMES:
            recognizers.append({
                "name": name,
                "label": {"lbph": "LBPH (baseline)",
                          "sface": "SFace (embeddings)"}.get(name, name),
                "trained": Path(default_model_path(name)).is_file(),
            })
        return {
            "detectors": list(DETECTOR_NAMES),
            "recognizers": recognizers,
            "yunet_available": config.YUNET_MODEL.is_file(),
        }

    @staticmethod
    def _files_payload() -> dict:
        videos, photos = [], []
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        for path in sorted(UPLOAD_DIR.iterdir()):
            if not path.is_file():
                continue
            entry = {"path": str(path), "label": f"uploads / {path.name}"}
            if path.suffix.lower() in config.IMAGE_EXTENSIONS:
                photos.append(entry)
            else:
                videos.append(entry)

        # Enrolled people are the handiest photo folders to replay.
        if config.DATASET_DIR.is_dir():
            for folder in sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir()):
                if any(f.suffix.lower() in config.IMAGE_EXTENSIONS for f in folder.iterdir()):
                    photos.append({"path": str(folder), "label": f"dataset / {folder.name}"})

        return {"videos": videos, "photos": photos}

    # -------------------------------------------------------------- actions
    def _start(self, body: dict) -> None:
        threshold = body.get("threshold")
        try:
            threshold = float(threshold) if threshold not in (None, "") else None
        except (TypeError, ValueError):
            self._json({"error": "threshold must be a number"}, 400)
            return

        self.session.stop()
        try:
            self.session.start(
                source=body.get("source") or None,
                camera_index=int(body.get("camera_index", config.CAMERA_INDEX)),
                detector=body.get("detector", "haar"),
                recognizer=body.get("recognizer", "none"),
                threshold=threshold,
                loop=bool(body.get("loop")),
                scale_factor=body.get("scale_factor", config.HAAR_SCALE_FACTOR),
                min_neighbors=body.get("min_neighbors", config.HAAR_MIN_NEIGHBORS),
                min_size=body.get("min_size", config.HAAR_MIN_SIZE[0]),
                score_threshold=body.get("score_threshold", config.YUNET_SCORE_THRESHOLD),
            )
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            return
        self._json({"ok": True, "detector_description": self.session.detector_description})

    def _upload(self, query: dict) -> None:
        name = (query.get("name") or ["upload.bin"])[0]
        safe = Path(name).name.replace("\\", "_")
        if not safe:
            self._json({"error": "bad file name"}, 400)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json({"error": "empty upload"}, 400)
            return
        if length > MAX_UPLOAD:
            self._json({"error": f"file larger than {MAX_UPLOAD // (1024 * 1024)} MB"}, 413)
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / safe
        remaining = length
        with open(target, "wb") as handle:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                remaining -= len(chunk)
        self._json({"ok": True, "name": safe, "path": str(target)})


# ====================================================================== serve
def local_ip() -> str:
    """Best guess at the address other devices on the Wi-Fi should use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))       # no traffic is sent, this just picks a route
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def serve(host: str = "0.0.0.0", port: int = 8000,
          session: Optional[PipelineSession] = None) -> int:
    config.ensure_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Handler.session = session or PipelineSession()

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

    shown = local_ip() if host in ("0.0.0.0", "") else host
    print("=" * 62)
    print("Face Track web interface")
    print("=" * 62)
    print(f"  on this machine : http://localhost:{port}/")
    if host in ("0.0.0.0", ""):
        print(f"  on the Wi-Fi    : http://{shown}:{port}/")
        print("  (open that address on a laptop or phone on the same network)")
    print("\npress Ctrl+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        Handler.session.stop()
        httpd.server_close()
    return 0
