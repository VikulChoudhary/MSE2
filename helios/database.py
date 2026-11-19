"""Lightweight durable experiment, prediction, model, and incident store."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json(value):
    return json.loads(value) if value else None


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS versions (
                version INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL, artifact_path TEXT NOT NULL,
                metrics TEXT NOT NULL, reference TEXT NOT NULL,
                training_rows INTEGER NOT NULL, source TEXT NOT NULL
              );
              CREATE TABLE IF NOT EXISTS deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                version INTEGER NOT NULL, previous_version INTEGER,
                reason TEXT NOT NULL
              );
              CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, version INTEGER NOT NULL,
                features TEXT NOT NULL, predicted INTEGER NOT NULL,
                confidence REAL NOT NULL, latency_ms REAL NOT NULL
              );
              CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(timestamp);
              CREATE TABLE IF NOT EXISTS feedback (
                prediction_id TEXT PRIMARY KEY REFERENCES predictions(id),
                timestamp TEXT NOT NULL, actual INTEGER NOT NULL
              );
              CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                source TEXT NOT NULL, parameters TEXT NOT NULL,
                metrics TEXT NOT NULL, accepted INTEGER NOT NULL,
                note TEXT NOT NULL, version INTEGER
              );
              CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, opened_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, category TEXT NOT NULL,
                status TEXT NOT NULL, root_cause TEXT NOT NULL,
                evidence TEXT NOT NULL, action TEXT NOT NULL DEFAULT 'none',
                action_at TEXT, verification TEXT
              );
              CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status, category);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create_version(self, artifact_path: str, metrics: dict, reference: dict, rows: int, source: str) -> int:
        with self.connect() as db:
            cur = db.execute("INSERT INTO versions (created_at,artifact_path,metrics,reference,training_rows,source) VALUES (?,?,?,?,?,?)",
                             (utc_now(), artifact_path, json.dumps(metrics), json.dumps(reference), rows, source))
            return int(cur.lastrowid)

    @staticmethod
    def version_dict(row):
        if row is None:
            return None
        out = dict(row)
        out["metrics"] = parse_json(out["metrics"])
        out["reference"] = parse_json(out["reference"])
        return out

    def version(self, num: int):
        with self.connect() as db:
            row = db.execute("SELECT * FROM versions WHERE version=?", (num,)).fetchone()
        return self.version_dict(row)

    def versions(self, limit: int = 50):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM versions ORDER BY version DESC LIMIT ?", (limit,)).fetchall()
        return [self.version_dict(row) for row in rows]

    def deployment(self, version: int, previous: int | None, reason: str):
        with self.connect() as db:
            db.execute("INSERT INTO deployments (timestamp,version,previous_version,reason) VALUES (?,?,?,?)",
                       (utc_now(), version, previous, reason))

    def previous_deployed_versions(self, active: int):
        with self.connect() as db:
            rows = db.execute("SELECT version FROM deployments WHERE version != ? ORDER BY id DESC", (active,)).fetchall()
        return list(dict.fromkeys(int(r["version"]) for r in rows))

    def add_prediction(self, features: list[float], prediction: int, confidence: float, latency_ms: float, version: int):
        prediction_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
                       (prediction_id, utc_now(), version, json.dumps(features), prediction, confidence, latency_ms))
        return prediction_id

    def add_feedback(self, prediction_id: str, actual: int):
        with self.connect() as db:
            exists = db.execute("SELECT id FROM predictions WHERE id=?", (prediction_id,)).fetchone()
            if not exists:
                raise LookupError("Prediction ID not found")
            try:
                db.execute("INSERT INTO feedback (prediction_id,timestamp,actual) VALUES (?,?,?)",
                           (prediction_id, utc_now(), actual))
            except sqlite3.IntegrityError as exc:
                raise ValueError("Feedback has already been recorded for this prediction") from exc

    def recent_predictions(self, limit: int):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM predictions ORDER BY timestamp DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        out = [dict(r) for r in rows]
        for item in out:
            item["features"] = parse_json(item["features"])
        return out

    def labeled(self, limit: int = 500, version: int | None = None):
        query = "SELECT p.*, f.actual FROM feedback f JOIN predictions p ON p.id = f.prediction_id"
        params = []
        if version is not None:
            query += " WHERE p.version=?"
            params.append(version)
        query += " ORDER BY f.timestamp DESC, f.rowid DESC LIMIT ?"
        params.append(limit)
        with self.connect() as db:
            rows = db.execute(query, params).fetchall()
        out = [dict(r) for r in rows]
        for item in out:
            item["features"] = parse_json(item["features"])
        return out

    def add_experiment(self, source: str, parameters: dict, metrics: dict, accepted: bool, note: str, version: int | None):
        with self.connect() as db:
            db.execute("INSERT INTO experiments (timestamp,source,parameters,metrics,accepted,note,version) VALUES (?,?,?,?,?,?,?)",
                       (utc_now(), source, json.dumps(parameters), json.dumps(metrics), int(accepted), note, version))

    def experiments(self, limit: int = 50):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "metrics": parse_json(r["metrics"]), "parameters": parse_json(r["parameters"]),
                 "accepted": bool(r["accepted"])} for r in rows]

    def find_unresolved_incident(self, category: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM incidents WHERE category=? AND status != 'resolved' ORDER BY id DESC LIMIT 1", (category,)).fetchone()
        return self.incident_dict(row)

    @staticmethod
    def incident_dict(row):
        if row is None:
            return None
        data = dict(row)
        data["evidence"] = parse_json(data["evidence"])
        data["verification"] = parse_json(data["verification"])
        return data

    def create_or_update_incident(self, category: str, root_cause: str, evidence: dict):
        existing = self.find_unresolved_incident(category)
        with self.connect() as db:
            if existing:
                db.execute("UPDATE incidents SET updated_at=?, root_cause=?, evidence=? WHERE id=?",
                           (utc_now(), root_cause, json.dumps(evidence), existing["id"]))
                return existing["id"]
            cur = db.execute("INSERT INTO incidents (opened_at,updated_at,category,status,root_cause,evidence) VALUES (?,?,?,?,?,?)",
                             (utc_now(), utc_now(), category, "open", root_cause, json.dumps(evidence)))
            return cur.lastrowid

    def update_incident(self, incident_id: int, status: str, action: str, verification: dict | None = None):
        with self.connect() as db:
            db.execute("UPDATE incidents SET updated_at=?,status=?,action=?,action_at=?,verification=? WHERE id=?",
                       (utc_now(), status, action, utc_now(), json.dumps(verification), incident_id))

    def resolve_incidents(self, categories: set[str], details: dict):
        if not categories:
            return
        with self.connect() as db:
            for category in categories:
                db.execute("UPDATE incidents SET status='resolved',updated_at=?,verification=? WHERE category=? AND status!='resolved'",
                           (utc_now(), json.dumps(details), category))

    def incidents(self, limit: int = 50):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self.incident_dict(row) for row in rows]
