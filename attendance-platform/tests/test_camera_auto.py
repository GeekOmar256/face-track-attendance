from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import DEFAULTS
from vision import Camera
from install_pi_service import service_unit, unit_quote


class CameraAutoTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((20, 30, 3), dtype=np.uint8)

    def test_auto_skips_unopened_usb_and_keeps_first_frame(self):
        failed, working = Mock(), Mock()
        failed.isOpened.return_value = False
        working.read.return_value = (True, self.frame)
        with patch.dict(sys.modules, {"picamera2": None}), patch.object(Camera, "usb_indexes", return_value=[0, 2]), patch("vision.cv2.VideoCapture", side_effect=[failed, working]) as opened:
            camera = Camera(DEFAULTS)
            self.assertEqual([call.args[0] for call in opened.call_args_list], [0, 2])
            self.assertEqual(camera.selected, {"kind": "usb", "index": 2})
            np.testing.assert_array_equal(camera.read(), self.frame)
            working.read.assert_called_once()
            failed.release.assert_called_once()
            camera.close(); camera.close()
            working.release.assert_called_once()

    def test_auto_uses_reported_pi_number_and_timed_frame_wait(self):
        picamera = Mock()
        picamera.wait.return_value = self.frame
        library = Mock()
        library.Picamera2.global_camera_info.return_value = [{"Num": 4, "Id": "/base/imx219", "Model": "imx219"}]
        library.Picamera2.return_value = picamera
        with patch.dict(sys.modules, {"picamera2": library}), patch("vision.cv2.VideoCapture") as usb:
            camera = Camera(DEFAULTS)
            library.Picamera2.assert_called_once_with(4)
            picamera.capture_array.assert_called_once_with(wait=False)
            self.assertEqual(picamera.wait.call_args.kwargs["timeout"], 5)
            np.testing.assert_array_equal(camera.read(), self.frame)
            usb.assert_not_called()
            camera.close()

    def test_dead_pi_falls_back_to_working_usb(self):
        picamera, usb = Mock(), Mock()
        picamera.wait.side_effect = TimeoutError("frame timeout")
        library = Mock()
        library.Picamera2.global_camera_info.return_value = [{"Num": 0, "Id": "/base/imx219"}]
        library.Picamera2.return_value = picamera
        usb.read.return_value = (True, self.frame)
        with patch.dict(sys.modules, {"picamera2": library}), patch.object(Camera, "usb_indexes", return_value=[2]), patch("vision.cv2.VideoCapture", return_value=usb):
            camera = Camera(DEFAULTS)
            self.assertEqual(camera.selected, {"kind": "usb", "index": 2})
            picamera.close.assert_called_once()
            camera.close()

    def test_explicit_pi_does_not_silently_select_usb(self):
        with patch.dict(sys.modules, {"picamera2": None}), patch("vision.cv2.VideoCapture") as usb:
            with self.assertRaisesRegex(RuntimeError, "Picamera2|picamera2"):
                Camera({**DEFAULTS, "camera": "pi"})
            usb.assert_not_called()

    def test_opened_device_without_frames_is_rejected(self):
        usb = Mock()
        usb.read.return_value = (False, None)
        with patch.dict(sys.modules, {"picamera2": None}), patch.object(Camera, "usb_indexes", return_value=[2]), patch("vision.cv2.VideoCapture", return_value=usb):
            with self.assertRaisesRegex(RuntimeError, "returned no frame"):
                Camera(DEFAULTS)
            usb.release.assert_called_once()

    def test_service_runs_as_user_and_restarts_at_boot(self):
        unit = service_unit(Path('/home/face/face-track/attendance-platform'), 'face', Path('/home/face/.config/face-track/service.env'), 8010, ['video'])
        self.assertIn('User=face', unit)
        self.assertIn('SupplementaryGroups=video', unit)
        self.assertIn('Restart=always', unit)
        self.assertIn('WantedBy=multi-user.target', unit)
        self.assertIn(' --host 0.0.0.0 --port 8010', unit)
        self.assertNotIn('123456', unit)

    def test_service_quoting_rejects_newlines_and_handles_specifiers(self):
        self.assertEqual(unit_quote('/home/face/a%/b'), '"/home/face/a%%/b"')
        self.assertEqual(unit_quote('/home/face/a%/b', False), '"/home/face/a%/b"')
        with self.assertRaises(ValueError):
            unit_quote('file\nExecStart=bad')


if __name__ == '__main__':
    unittest.main()
