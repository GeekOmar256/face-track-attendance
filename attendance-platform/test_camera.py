import argparse
from pathlib import Path
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description="Standalone Pi/USB camera test; no attendance or Firebase.")
    parser.add_argument("--camera", choices=("pi", "usb"), default="pi")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true", help="Use over SSH without a desktop")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, default=Path("camera-test.jpg"))
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")

    picamera = None
    capture = None
    try:
        if args.camera == "pi":
            from picamera2 import Picamera2

            cameras = Picamera2.global_camera_info()
            print("Pi cameras:", cameras)
            if not 0 <= args.index < len(cameras):
                raise RuntimeError("No Pi camera at this index; check rpicam-hello --list-cameras")
            picamera = Picamera2(args.index)
            picamera.configure(picamera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}))
            picamera.start()
            time.sleep(2)
        else:
            capture = cv2.VideoCapture(args.index)
            if not capture.isOpened():
                raise RuntimeError("USB camera could not open; check its index and whether another app uses it")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        started = time.monotonic()
        frames = 0
        latest = None
        print("Testing camera. Press Q or Escape to stop the preview; Ctrl+C to stop over SSH.")
        try:
            while time.monotonic() - started < args.seconds:
                if picamera is not None:
                    frame = picamera.capture_array("main")
                else:
                    success, frame = capture.read()
                    if not success:
                        raise RuntimeError("Camera opened but failed to return a frame")
                if frame is None or frame.size == 0:
                    raise RuntimeError("Camera returned an empty frame")
                latest = frame
                frames += 1
                if not args.no_preview:
                    cv2.imshow("Camera test - Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        except KeyboardInterrupt:
            print("Stopped by user.")

        elapsed = time.monotonic() - started
        if latest is None:
            raise RuntimeError("No frames received")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), latest):
            raise RuntimeError("Could not save test image")
        height, width = latest.shape[:2]
        print(f"PASS: {frames} frames, {width}x{height}, {frames / max(elapsed, 0.001):.1f} acquisition FPS")
        print("Saved:", args.output.resolve())
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        print("Stop attendance/preview first. Over SSH, add --no-preview.")
        return 1
    finally:
        if picamera is not None:
            picamera.close()
        if capture is not None:
            capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
