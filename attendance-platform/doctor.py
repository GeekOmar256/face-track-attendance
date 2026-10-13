import importlib
import json
import platform


if __name__ == "__main__":
    print("Python", platform.python_version(), "on", platform.platform())
    for name in ("cv2", "numpy", "flask", "waitress", "openpyxl", "PIL", "dlib", "picamera2", "firebase_admin"):
        try:
            module = importlib.import_module(name)
            print(name, "OK", getattr(module, "__version__", ""))
        except Exception as exc:
            print(name, "UNAVAILABLE:", exc)
    try:
        from engine import DEFAULTS
        from vision import Detector, capabilities
        print(json.dumps(capabilities(), indent=2))
        Detector("haar", DEFAULTS)
        print("Haar cascade path: OK")
    except Exception as exc:
        print("Vision configuration:", exc)
