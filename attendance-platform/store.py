import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, class_name TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    image BLOB NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, class_name TEXT NOT NULL,
                    started TEXT NOT NULL, ended TEXT, state TEXT NOT NULL,
                    error TEXT, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS frames (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    timestamp TEXT NOT NULL, frame_number INTEGER NOT NULL,
                    gate_ms REAL NOT NULL, detect_ms REAL NOT NULL, recognition_ms REAL NOT NULL,
                    pipeline_ms REAL NOT NULL, write_ms REAL NOT NULL DEFAULT 0,
                    fps REAL NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                    gate_faces INTEGER NOT NULL, faces INTEGER NOT NULL, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, frame_id INTEGER NOT NULL REFERENCES frames(id),
                    session_id TEXT NOT NULL REFERENCES sessions(id), timestamp TEXT NOT NULL,
                    user_id TEXT, user_name TEXT, class_name TEXT, detector TEXT NOT NULL,
                    recognizer TEXT NOT NULL, box TEXT NOT NULL, detector_score REAL,
                    match_score REAL, accepted INTEGER NOT NULL, recognition_ms REAL NOT NULL,
                    thumbnail BLOB NOT NULL, image BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    user_id TEXT NOT NULL, name TEXT NOT NULL, class_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL, event_id INTEGER NOT NULL REFERENCES events(id),
                    UNIQUE(session_id, user_id));
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY, collection TEXT NOT NULL, document TEXT NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, result TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS frame_session ON frames(session_id);
                CREATE INDEX IF NOT EXISTS event_session ON events(session_id);
            """)
            interrupted = list(db.execute("SELECT id FROM sessions WHERE state='running'"))
            ended = now()
            db.execute("UPDATE sessions SET state='interrupted', ended=? WHERE state='running'", (ended,))
            for row in interrupted:
                self.enqueue("sessions", row["id"], {"state": "interrupted", "ended": ended}, db)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def query(self, sql, params=()):
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, params)]

    def execute(self, sql, params=()):
        with self.connect() as db:
            return db.execute(sql, params).lastrowid

    def enqueue(self, collection, document, payload, db=None):
        if db is None:
            with self.connect() as db:
                self.enqueue(collection, document, payload, db)
            return
        db.execute("INSERT INTO outbox(collection,document,payload) VALUES (?,?,?)",
                   (collection, str(document), json.dumps(payload, allow_nan=False)))

    def users(self):
        return self.query("""SELECT u.*, COUNT(p.id) AS photos FROM users u
                            LEFT JOIN photos p ON p.user_id=u.id WHERE u.active=1 GROUP BY u.id""")

    def history(self):
        return self.query("SELECT * FROM sessions ORDER BY started DESC LIMIT 200")

    def attendance(self, session_id):
        return self.query("""SELECT a.*, e.detector, e.recognizer, e.match_score, e.box,
                            e.detector_score FROM attendance a JOIN events e ON a.event_id=e.id
                            WHERE a.session_id=? ORDER BY a.timestamp""", (session_id,))

    def statistics(self, session_id):
        return self.query("""SELECT COUNT(*) AS frames, SUM(faces) AS faces,
            AVG(gate_ms) AS gate_ms, AVG(detect_ms) AS detection_ms,
            AVG(recognition_ms) AS recognition_ms, AVG(pipeline_ms) AS pipeline_ms,
            MAX(pipeline_ms) AS max_pipeline_ms, AVG(write_ms) AS write_ms,
            AVG(fps) AS mean_processing_fps,
            100.0*SUM(CASE WHEN faces>0 THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) AS frames_with_face_percent
            FROM frames WHERE session_id=?""", (session_id,))[0]

    def recognized_people(self, session_id):
        return self.query("""SELECT e.id,e.frame_id,e.timestamp,e.user_id,e.user_name,e.detector,
            e.recognizer,e.box,e.detector_score,e.match_score,e.accepted,e.recognition_ms
            FROM events e JOIN (
                SELECT MIN(id) AS first_event FROM events
                WHERE session_id=? AND accepted=1 AND user_id IS NOT NULL
                GROUP BY user_id
            ) people ON people.first_event=e.id ORDER BY e.id""", (session_id,))
