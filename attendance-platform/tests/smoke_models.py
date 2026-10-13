import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import Engine
from store import Store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise real models using an existing enrollment dataset")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    photos = [photo for photo in sorted(args.dataset.glob("*/*")) if photo.suffix.lower() in (".jpg", ".png")]
    if not photos:
        raise SystemExit("No dataset images")
    examples = {}
    for photo in photos:
        examples.setdefault(photo.parent.name, photo)
    with tempfile.TemporaryDirectory() as directory:
        store = Store(Path(directory) / "smoke.sqlite3")
        engine = Engine(store)
        for index, photo in enumerate(examples.values()):
            engine.add_user({"id": f"test{index}", "name": f"Test person {index}", "class_name": "Test"}, [photo.read_bytes()])
        results = []
        for detector, recognizer in (("haar", "lbph"), ("yunet", "sface"), ("dlib", "dlib")):
            engine.save_settings({"detector": detector, "recognizer": recognizer, "gate_enabled": False})
            session = engine.start({"name": f"{detector}/{recognizer}", "mode": "single", "source_path": str(photos[0])})
            deadline = time.monotonic()+90
            while engine.running() and time.monotonic() < deadline:
                time.sleep(0.1)
            if engine.running():
                raise RuntimeError("Model test timed out")
            if engine.error:
                raise RuntimeError(engine.error)
            stats = store.statistics(session["session_id"])
            if stats["frames"] != 1:
                raise RuntimeError("Expected exactly one processed image")
            results.append({"detector": detector, "recognizer": recognizer, "statistics": stats,
                            "attendance": len(store.attendance(session["session_id"]))})
        print(json.dumps(results, indent=2))
        print("Smoke test only: enrollment and probe share a source; this is not accuracy evaluation.")
