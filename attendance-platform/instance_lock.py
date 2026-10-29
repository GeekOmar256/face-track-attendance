"""Keep two attendance processes from sharing a database and camera."""
import os
from pathlib import Path


class InstanceLock:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def acquire(self):
        if self.stream is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if stream.seek(0, os.SEEK_END) == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError("Face Track is already running for this installation. Use the existing "
                               "service/local page; do not also launch python app.py. "
                               "On Pi run: systemctl status face-track --no-pager") from exc
        self.stream = stream
        return self

    def release(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            stream.close()  # Closing releases the OS lock, including after process termination.

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()
