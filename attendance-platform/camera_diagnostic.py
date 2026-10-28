import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse


PAGE = """<!doctype html><html lang="en"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera diagnostic</title><style>
body{font-family:system-ui;background:#eef3f8;color:#203348;margin:0;padding:20px}
main{max-width:850px;margin:auto}section{background:white;border-radius:14px;padding:20px;margin:16px 0}
img{width:100%;max-height:65vh;object-fit:contain;background:#152536;border-radius:10px}
pre{white-space:pre-wrap;overflow-wrap:anywhere}p{line-height:1.5}
</style><main><h1>Camera diagnostic</h1><p>Independent camera test. No attendance or Firebase connection.</p>
<section><h2 id="state">Starting</h2><p id="stats"></p><img id="preview" alt="Waiting for camera frames"></section>
<section><h2>Results</h2><div id="results"></div></section>
<details><summary>System checks and discovery</summary><pre id="checks"></pre></details></main>
<script>
async function update(){try{
 const response=await fetch('/status',{cache:'no-store'});const report=await response.json();
 document.getElementById('state').textContent=report.state;
 const current=report.current||{};const stats=current.statistics||{};
 document.getElementById('stats').textContent=(current.label||'')+' | Frames: '+(stats.frames||0)+' | FPS: '+(stats.fps||0);
 if(stats.frames){document.getElementById('preview').src='/frame?t='+Date.now();}
 document.getElementById('results').replaceChildren();
 for(const result of report.results){const paragraph=document.createElement('p');
 paragraph.textContent=result.label+': '+result.result+' — '+result.advice.join(' ');
 document.getElementById('results').append(paragraph);}
 document.getElementById('checks').textContent=JSON.stringify({cameras:report.cameras,checks:report.checks},null,2);
 }catch(error){document.getElementById('state').textContent='Connection lost; check terminal.';}}
 update();setInterval(update,500);
</script></html>"""


def run_check(command, timeout=6):
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
        return {"exit_code": result.returncode, "output": (result.stdout + result.stderr)[-12000:]}
    except subprocess.TimeoutExpired as exc:
        captured = (exc.stdout or b"") + (exc.stderr or b"")
        if isinstance(captured, bytes):
            captured = captured.decode(errors="replace")
        return {"exit_code": None, "output": (captured + "\n" + str(exc))[-12000:]}
    except OSError as exc:
        return {"exit_code": None, "output": str(exc)}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def discover():
    checks = {"system": {"os": platform.platform(), "python": sys.version, "interpreter": sys.executable}}
    cameras = []
    probe = run_check([sys.executable, str(Path(__file__).resolve()), "--worker", "discover"], 12)
    checks["picamera2_discovery"] = probe
    for line in probe["output"].splitlines():
        if line.startswith("CAMERAS_JSON="):
            for position, camera in enumerate(json.loads(line.split("=", 1)[1])):
                index = camera.get("Num", position)
                cameras.append({"kind": "pi", "index": index, "label": f"Pi {index}: {camera.get('Model', 'camera')}"})
    if platform.system() == "Linux":
        for name, command in (
            ("rpicam_list", ["rpicam-hello", "--list-cameras"]),
            ("power", ["vcgencmd", "get_throttled"]),
            ("v4l2_list", ["v4l2-ctl", "--list-devices"]),
            ("kernel", ["dmesg"]),
        ):
            checks[name] = run_check(command)
        if not cameras:
            for match in re.finditer(r"^\s*(\d+)\s*:\s*(\S+)\s*\[", checks["rpicam_list"]["output"], re.MULTILINE):
                cameras.append({"kind": "pi", "index": int(match.group(1)), "label": f"Pi {match.group(1)}: {match.group(2)} (native discovery)"})
        for filename in ("/etc/os-release", "/boot/firmware/config.txt"):
            try:
                lines = Path(filename).read_text().splitlines()
                if filename.endswith("config.txt"):
                    lines = [line for line in lines if re.search(r"camera_auto_detect|dtoverlay=.*imx", line)]
                checks[filename] = "\n".join(lines)
            except OSError as exc:
                checks[filename] = str(exc)
        for device in sorted(Path("/sys/class/video4linux").glob("video*")):
            if "/usb" not in str(device.resolve()):
                continue
            node = "/dev/" + device.name
            capability = run_check(["v4l2-ctl", "--device", node, "--all"])
            checks[node] = capability
            if capability["exit_code"] != 0 or not re.search(r"Video Capture", capability["output"]):
                continue
            label = device.name
            try:
                label = (device / "name").read_text().strip()
            except OSError:
                pass
            cameras.append({"kind": "usb", "index": int(device.name[5:]), "label": f"USB {node}: {label}"})
    else:
        cameras.extend({"kind": "usb", "index": index, "label": f"Webcam index {index} (unverified)"} for index in range(4))
    return checks, cameras


def diagnose(log, result, frames):
    lowered = log.lower()
    if result == "PASS":
        return ["Frames received and snapshot saved. This verifies capture, not face recognition."]
    if "no module named" in lowered:
        return ["Missing Python dependency. On Pi install python3-opencv/python3-picamera2 with apt and use a --system-site-packages venv."]
    if "permission denied" in lowered or "operation not permitted" in lowered:
        return ["Device permission problem: check video-group membership and device permissions; do not run the whole app as root."]
    if any(message in lowered for message in ("device or resource busy", "pipeline handler in use", "failed to acquire camera")):
        return ["Camera may be occupied. Stop attendance, its preview, and other camera programs, then retry."]
    if "could not open" in lowered:
        return ["Device could not open: check index, USB connection, permissions, and other camera processes."]
    if result == "TIMEOUT" or "timed out" in lowered or "timeout" in lowered or "failed to return" in lowered:
        return [f"Capture stalled after {frames} frames. A detected sensor does not prove frames can transfer.",
                "Stop other camera programs and compare with rpicam-still --nopreview --timeout 3000 -o ~/camera-check.jpg.",
                "If the native test also fails, investigate cable/connector, camera, power, and OS/driver compatibility; this test cannot identify a broken component conclusively."]
    return ["See the saved worker log. Check camera availability and Python dependencies; the exact cause is not established."]


def capture_worker(args):
    import cv2

    folder = args.worker_dir
    picamera = None
    capture = None
    try:
        if args.camera == "pi":
            from picamera2 import Picamera2

            picamera = Picamera2(args.index)
            picamera.configure(picamera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}, controls={"FrameRate": 15}))
            picamera.start()
        else:
            backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
            capture = cv2.VideoCapture(args.index, backend)
            if not capture.isOpened():
                raise RuntimeError("Camera could not open")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        started = time.monotonic()
        frames = 0
        last_preview = 0
        while time.monotonic() - started < args.seconds:
            if picamera is not None:
                frame = picamera.capture_array("main")
            else:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Camera failed to return a frame")
            if frame is None or not frame.size:
                raise RuntimeError("Empty frame")
            frames += 1
            elapsed = time.monotonic() - started
            if elapsed - last_preview >= 0.2 or frames == 1:
                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    raise RuntimeError("JPEG encoding failed")
                temporary = folder / "frame.tmp"
                temporary.write_bytes(encoded.tobytes())
                temporary.replace(folder / "frame.jpg")
                write_json(folder / "statistics.json", {"frames": frames, "fps": round(frames / max(elapsed, 0.001), 1),
                           "width": frame.shape[1], "height": frame.shape[0]})
                last_preview = elapsed
        write_json(folder / "statistics.json", {"frames": frames, "fps": round(frames / max(elapsed, 0.001), 1),
                   "width": frame.shape[1], "height": frame.shape[0]})
    finally:
        if picamera is not None:
            picamera.close()
        if capture is not None:
            capture.release()


class CameraTest:
    def __init__(self, args, checks, cameras):
        self.args = args
        self.folder = args.output.resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.report = {"state": "Ready", "checks": checks, "cameras": cameras, "current": {}, "results": []}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.report))

    def run(self):
        try:
            for camera in self.report["cameras"]:
                if self.stop.is_set():
                    break
                self.test(camera)
            with self.lock:
                self.report["state"] = "Stopped" if self.stop.is_set() else "Complete" if self.report["cameras"] else "No cameras found"
        except Exception as exc:
            with self.lock:
                self.report["state"] = f"Diagnostic error: {exc}"
        finally:
            write_json(self.folder / "report.json", self.snapshot())
            print("Report:", self.folder / "report.json", flush=True)

    def test(self, camera):
        folder = self.folder / f"{camera['kind']}-{camera['index']}"
        folder.mkdir(exist_ok=True)
        with self.lock:
            self.report["state"] = "Testing"
            self.report["current"] = {**camera, "statistics": {}}
            self.frame = None
        print("Testing", camera["label"], flush=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "capture", "--camera", camera["kind"],
                   "--index", str(camera["index"]), "--seconds", str(self.args.seconds), "--worker-dir", str(folder)]
        result = "FAIL"
        statistics = {}
        with (folder / "worker.log").open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            started = time.monotonic()
            last_progress = started
            try:
                while process.poll() is None:
                    latest = read_json(folder / "statistics.json")
                    if latest.get("frames", 0) > statistics.get("frames", 0):
                        last_progress = time.monotonic()
                        statistics = latest
                        with self.lock:
                            self.frame = folder / "frame.jpg"
                            self.report["current"]["statistics"] = statistics
                    if self.stop.is_set():
                        result = "STOPPED"
                        break
                    if time.monotonic() - last_progress > self.args.timeout or time.monotonic() - started > self.args.seconds + 2 * self.args.timeout:
                        result = "TIMEOUT"
                        break
                    self.stop.wait(0.1)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            statistics = read_json(folder / "statistics.json") or statistics
            if result not in ("TIMEOUT", "STOPPED") and process.returncode == 0 and statistics.get("frames", 0):
                result = "PASS"
        log = (folder / "worker.log").read_text(encoding="utf-8", errors="replace")
        advice = diagnose(log, result, statistics.get("frames", 0))
        native = None
        if camera["kind"] == "pi" and result not in ("PASS", "STOPPED") and not self.stop.is_set() and shutil.which("rpicam-still"):
            print("Comparing with native rpicam-still (bounded to 8 seconds)...", flush=True)
            native = run_check(["rpicam-still", "--camera", str(camera["index"]), "--nopreview", "--timeout", "2000",
                                "-o", str(folder / "native-check.jpg")], 8)
            if native["exit_code"] == 0 and (folder / "native-check.jpg").exists():
                advice = ["Native rpicam capture succeeded, but Python capture failed. Investigate Picamera2/OpenCV environment or camera configuration; hardware captured successfully in the native test."]
            else:
                advice.append("Native rpicam capture also failed; the failure is not limited to this Python capture code. See native_check in the report.")
        if result == "STOPPED":
            advice = ["Stopped by user."]
        with self.lock:
            self.report["current"]["statistics"] = statistics
            if (folder / "frame.jpg").exists():
                self.frame = folder / "frame.jpg"
            self.report["results"].append({**camera, "result": result, "statistics": statistics,
                                          "advice": advice, "log": log[-12000:], "native_check": native})
        print(result + ": " + " ".join(advice), flush=True)
        write_json(self.folder / "report.json", self.snapshot())


def preview_server(test, args):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                payload, content_type = PAGE.encode(), "text/html; charset=utf-8"
            elif route == "/status":
                payload, content_type = json.dumps(test.snapshot()).encode(), "application/json"
            elif route == "/frame":
                try:
                    with test.lock:
                        frame = test.frame
                    if frame is None:
                        raise FileNotFoundError()
                    payload, content_type = frame.read_bytes(), "image/jpeg"
                except OSError:
                    self.send_error(404, "No frame yet")
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *messages):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    thread = threading.Thread(target=test.run)
    thread.start()
    print(f"Preview: http://localhost:{server.server_port} on this device; use http://PI_IP:{server.server_port} from another device.", flush=True)
    print("Read-only preview has no login. Use a trusted LAN only. Ctrl+C closes it and releases the test camera.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        test.stop.set()
        thread.join()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Discover cameras, diagnose capture and preview in a browser, including over SSH.")
    parser.add_argument("--camera", choices=("auto", "pi", "usb"), default="auto")
    parser.add_argument("--index", type=int)
    parser.add_argument("--list", action="store_true", help="Discovery only; no capture")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=15, help="Maximum wait for first/next frame")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--output", type=Path, default=Path("data") / ("camera-diagnostic-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    parser.add_argument("--worker", choices=("discover", "capture"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.seconds <= 0 or args.timeout <= 0 or (args.index is not None and args.index < 0):
        parser.error("Duration/timeout must be positive and index must be nonnegative")
    if args.worker == "discover":
        from picamera2 import Picamera2

        print("CAMERAS_JSON=" + json.dumps(Picamera2.global_camera_info(), default=str))
        return 0
    if args.worker == "capture":
        capture_worker(args)
        return 0
    print("Stop attendance, its camera preview and other camera programs before testing.", flush=True)
    checks, cameras = discover()
    if args.camera != "auto":
        cameras = [camera for camera in cameras if camera["kind"] == args.camera]
    if args.index is not None:
        cameras = [camera for camera in cameras if camera["index"] == args.index]
        if not cameras and args.camera != "auto":
            cameras = [{"kind": args.camera, "index": args.index, "label": f"Requested {args.camera} camera {args.index}"}]
    print("Cameras:", json.dumps(cameras, indent=2))
    power = checks.get("power", {}).get("output", "")
    match = re.search(r"throttled=(0x[0-9a-fA-F]+)", power)
    if match:
        flags = int(match.group(1), 16)
        print("Power: " + ("undervoltage flag detected (current or since boot)" if flags & 0x10001 else "no undervoltage flag reported"))
    test = CameraTest(args, checks, cameras)
    if args.list:
        write_json(test.folder / "report.json", test.snapshot())
        print("Discovery report:", test.folder / "report.json")
        return 0
    if not cameras:
        print("No capture candidates found. Check report.json; install v4l-utils for USB discovery or python3-picamera2 for Pi discovery.")
        test.run()
        return 1
    if args.no_preview:
        try:
            test.run()
        except KeyboardInterrupt:
            test.stop.set()
            write_json(test.folder / "report.json", test.snapshot())
    else:
        preview_server(test, args)
    return 0 if any(result["result"] == "PASS" for result in test.report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
