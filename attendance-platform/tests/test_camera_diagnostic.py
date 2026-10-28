import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import camera_diagnostic


class CameraDiagnosticTests(unittest.TestCase):
    def test_missing_dependency(self):
        advice = camera_diagnostic.diagnose("ModuleNotFoundError: No module named 'picamera2'", "FAIL", 0)
        self.assertIn("Missing Python dependency", advice[0])

    def test_busy_camera(self):
        advice = camera_diagnostic.diagnose("Failed to acquire camera", "FAIL", 0)
        self.assertIn("occupied", advice[0])

    def test_timeout_is_not_conclusive_hardware_diagnosis(self):
        advice = camera_diagnostic.diagnose("Camera frontend has timed out", "TIMEOUT", 0)
        self.assertIn("cannot identify a broken component conclusively", " ".join(advice))

    def test_discovery_parses_camera_numbers(self):
        probe = {"exit_code": 0, "output": 'CAMERAS_JSON=[{"Num": 2, "Model": "imx219"}]'}
        with patch.object(camera_diagnostic, "run_check", return_value=probe), patch.object(camera_diagnostic.platform, "system", return_value="Other"):
            checks, cameras = camera_diagnostic.discover()
        self.assertEqual(cameras[0]["index"], 2)
        self.assertEqual(cameras[0]["kind"], "pi")

    def test_missing_json_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(camera_diagnostic.read_json(Path(directory) / "missing.json"), {})

    def test_subprocess_timeout_preserves_error_log(self):
        import subprocess

        error = subprocess.TimeoutExpired(["rpicam-still"], 1, output=b"Camera frontend has timed out")
        with patch.object(camera_diagnostic.subprocess, "run", side_effect=error):
            check = camera_diagnostic.run_check(["rpicam-still"], 1)
        self.assertIn("Camera frontend has timed out", check["output"])
        self.assertIsNone(check["exit_code"])

    def test_successful_worker_statistics_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), seconds=1, timeout=1)
            test = camera_diagnostic.CameraTest(args, {}, [])
            def completed_worker(command, **kwargs):
                folder = Path(command[-1])
                camera_diagnostic.write_json(folder / "statistics.json", {"frames": 15, "fps": 15})
                (folder / "frame.jpg").write_bytes(b"test-image")
                from unittest.mock import Mock

                return Mock(returncode=0, poll=lambda: 0)

            with patch.object(camera_diagnostic.subprocess, "Popen", side_effect=completed_worker):
                test.test({"kind": "usb", "index": 0, "label": "Test webcam"})
            self.assertEqual(test.report["results"][0]["result"], "PASS")
            self.assertEqual(test.report["current"]["statistics"]["frames"], 15)
            self.assertTrue(test.frame.exists())

    def test_empty_discovery_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory))
            test = camera_diagnostic.CameraTest(args, {}, [])
            test.run()
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(report["state"], "No cameras found")

    def test_browser_routes_without_camera_hardware(self):
        import threading
        from urllib.request import urlopen

        responses = []
        errors = []
        original_server = camera_diagnostic.ThreadingHTTPServer

        class TestServer(original_server):
            def serve_forever(self):
                self.timeout = 3
                def request_routes():
                    try:
                        for route in ("/", "/status", "/frame"):
                            with urlopen(f"http://127.0.0.1:{self.server_port}{route}", timeout=3) as response:
                                responses.append(response.read())
                    except Exception as exc:
                        errors.append(exc)

                requester = threading.Thread(target=request_routes)
                requester.start()
                for request in range(3):
                    self.handle_request()
                requester.join(timeout=4)
                raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), host="127.0.0.1", port=0)
            test = camera_diagnostic.CameraTest(args, {}, [])
            test.frame = Path(directory) / "frame.jpg"
            test.frame.write_bytes(b"sample-preview")
            with patch.object(camera_diagnostic, "ThreadingHTTPServer", TestServer):
                camera_diagnostic.preview_server(test, args)
        self.assertFalse(errors)
        self.assertIn(b"Camera diagnostic", responses[0])
        self.assertIn("results", json.loads(responses[1]))
        self.assertEqual(responses[2], b"sample-preview")

    def test_stalled_worker_is_terminated(self):
        class StalledProcess:
            def __init__(self):
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout):
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory), seconds=1, timeout=0.01)
            test = camera_diagnostic.CameraTest(args, {}, [])
            process = StalledProcess()
            with patch.object(camera_diagnostic.subprocess, "Popen", return_value=process):
                test.test({"kind": "usb", "index": 0, "label": "Test webcam"})
            self.assertTrue(process.terminated)
            self.assertEqual(test.report["results"][0]["result"], "TIMEOUT")


if __name__ == "__main__":
    unittest.main()
