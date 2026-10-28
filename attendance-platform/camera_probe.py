"""Exercise the application's actual automatic camera selection without starting attendance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from engine import DEFAULTS
from vision import Camera, jpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--report", type=Path, default=Path("data/camera-probe.json"))
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("Seconds must be positive")
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "ok": False}
    camera = None
    try:
        camera = Camera({**DEFAULTS, "camera": "auto"})
        started, frames = time.monotonic(), 0
        while time.monotonic() - started < args.seconds:
            frame = camera.read()
            if frame is None:
                raise RuntimeError("Camera stopped returning frames")
            frames += 1
            if frames == 1 and args.snapshot:
                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                args.snapshot.write_bytes(jpeg(frame))
        result.update(ok=True, selected=camera.selected, frames=frames,
                      fps=round(frames / max(time.monotonic()-started, .001), 2))
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if camera:
            camera.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
