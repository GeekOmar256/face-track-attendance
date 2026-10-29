from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from instance_lock import InstanceLock
from restart_pi_service import belongs_to_installation


class InstanceTests(unittest.TestCase):
    def test_duplicate_is_rejected_and_release_allows_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = InstanceLock(Path(folder) / "app.lock"), InstanceLock(Path(folder) / "app.lock")
            with first:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    second.acquire()
            with second:
                self.assertIsNotNone(second.stream)

    def test_another_process_cannot_take_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "app.lock"
            code = "from instance_lock import InstanceLock; import sys; InstanceLock(sys.argv[1]).acquire()"
            with InstanceLock(path):
                blocked = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT,
                                         text=True, capture_output=True, timeout=10)
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn("already running", blocked.stderr)
            allowed = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT,
                                     text=True, capture_output=True, timeout=10)
            self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_duplicate_cleanup_matches_only_this_installation(self):
        root = ROOT.resolve()
        self.assertTrue(belongs_to_installation(["/usr/bin/python3", "app.py"], root, root))
        self.assertTrue(belongs_to_installation(["python", "-u", str(root / "app.py")], root.parent, root))
        self.assertFalse(belongs_to_installation(["python", "app.py"], root.parent, root))
        self.assertFalse(belongs_to_installation(["python", "other.py"], root, root))
        self.assertFalse(belongs_to_installation(["bash", "app.py"], root, root))
        self.assertFalse(belongs_to_installation(["python", "app.py"], None, root))


if __name__ == "__main__":
    unittest.main()
