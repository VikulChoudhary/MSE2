"""Versioned local model registry with atomic champion pointer and rollback."""
import json
import os
from pathlib import Path
from threading import RLock
from .training_model import FraudModel

from .database import Store


class Registry:
    def __init__(self, model_dir: Path, store: Store):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.pointer = self.model_dir / "champion.json"
        self.lock = RLock()
        self._cached = None
        self._cached_version = None

    def active_version(self) -> int | None:
        if not self.pointer.exists():
            return None
        try:
            return int(json.loads(self.pointer.read_text(encoding="utf-8"))["version"])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            raise RuntimeError("Champion pointer is damaged") from exc

    def load_version(self, version: int):
        metadata = self.store.version(version)
        if metadata is None:
            raise LookupError(f"Unknown model version {version}")
        # File locations are generated internally, never accepted from user requests.
        path = self.model_dir / f"model_v{version}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact missing for version {version}")
        return FraudModel.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def active_model(self):
        with self.lock:
            version = self.active_version()
            if version is None:
                raise RuntimeError("No champion model has been deployed")
            if self._cached_version != version:
                self._cached = self.load_version(version)
                self._cached_version = version
            return self._cached, version

    def register(self, model, metrics: dict, reference: dict, rows: int, source: str) -> int:
        with self.lock:
            # Artifact names are derived only from the locally allocated version ID.
            # If serialization fails, the metadata row remains unused, not deployable.
            version = self.store.create_version("pending", metrics, reference, rows, source)
            target = self.model_dir / f"model_v{version}.json"
            temp = self.model_dir / f"model_v{version}.tmp"
            try:
                temp.write_text(json.dumps(model.to_dict()), encoding="utf-8")
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
            with self.store.connect() as db:
                db.execute("UPDATE versions SET artifact_path=? WHERE version=?", (target.name, version))
            return version

    def activate(self, version: int, reason: str):
        with self.lock:
            # Do not activate anything that cannot be deserialized.
            self.load_version(version)
            try:
                previous = self.active_version()
            except RuntimeError:
                previous = None
            temp = self.pointer.with_suffix(".tmp")
            temp.write_text(json.dumps({"version": version}), encoding="utf-8")
            os.replace(temp, self.pointer)
            self._cached = None
            self._cached_version = None
            self.store.deployment(version, previous, reason)
            return {"previous_version": previous, "active_version": version, "reason": reason}

    def rollback(self):
        active = self.active_version()
        if active is None:
            raise RuntimeError("No active model to roll back")
        for version in self.store.previous_deployed_versions(active):
            try:
                self.load_version(version)
            except Exception:
                continue
            return self.activate(version, "automatic or requested rollback")
        raise RuntimeError("No earlier healthy deployed artifact available for rollback")

    def models(self):
        active = self.active_version()
        return {"champion_version": active,
                "versions": [{**v, "is_champion": v["version"] == active} for v in self.store.versions()]}
