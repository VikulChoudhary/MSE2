"""Orchestration and public service methods."""
import logging
from threading import Event, Thread, RLock
from time import perf_counter

from prometheus_client import Counter, Histogram

from .config import Settings
from .database import Store
from .monitoring import Monitor
from .recovery import RecoveryEngine
from .registry import Registry
from .training import CLASS_NAMES, Trainer

LOG = logging.getLogger(__name__)
PREDICTIONS = Counter("helios_predictions_total", "Number of predictions served", ["version", "class_name"])
LATENCY = Histogram("helios_prediction_latency_seconds", "Prediction latency in seconds")


class HeliosService:
    def __init__(self, settings: Settings):
        settings.validate()
        self.settings = settings
        self.store = Store(settings.db_path)
        self.registry = Registry(settings.model_dir, self.store)
        self.trainer = Trainer(settings, self.store, self.registry)
        self.monitor = Monitor(settings, self.store, self.registry)
        self.recovery = RecoveryEngine(settings, self.store, self.registry, self.monitor, self.trainer)
        self._stop = Event()
        self._worker = None
        self._predict_lock = RLock()

    def bootstrap(self):
        """Start reliably, including recovering from a missing/corrupt champion."""
        try:
            active = self.registry.active_version()
        except RuntimeError:
            # Keep a forensic copy of a damaged pointer, never load it as JSON again.
            damaged = self.registry.model_dir / "champion.damaged.json"
            self.registry.pointer.replace(damaged)
            active = None
        if active is not None:
            try:
                self.registry.active_model()
                return
            except Exception as exc:
                LOG.warning("Active model cannot load at startup: %s", exc)
                for candidate in self.store.previous_deployed_versions(active):
                    if self._recover_version(candidate):
                        return
        else:
            # Reuse an existing quality-gated artifact if pointer is lost.
            for candidate in self.store.versions():
                if self._recover_version(candidate["version"]):
                    return
        # A clean installation (or no usable backup) builds a fresh baseline.
        initial = self.trainer.train(source="bootstrap", use_feedback=False)
        if not initial["deployed"]:
            raise RuntimeError("Baseline model failed configured quality gates")
        self.registry.active_model()

    def _recover_version(self, version: int) -> bool:
        try:
            evaluation = self.trainer.evaluate_version(version)
            if (evaluation["accuracy"] < self.settings.candidate_accuracy_threshold or
                    evaluation["fraud_recall"] < self.settings.min_fraud_recall):
                return False
            self.registry.activate(version, reason="verified startup recovery")
            LOG.warning("Recovered champion version %s at startup", version)
            return True
        except Exception as exc:
            LOG.warning("Recovery candidate v%s is not usable: %s", version, exc)
            return False

    def predict(self, features: list[float]):
        with self._predict_lock:
            start = perf_counter()
            model, version = self.registry.active_model()
            prediction = int(model.predict([features])[0])
            probabilities = model.predict_proba([features])[0]
            confidence = float(probabilities[prediction])
            fraud_probability = float(probabilities[1])
            elapsed = (perf_counter() - start)
            latency_ms = elapsed * 1000
            prediction_id = self.store.add_prediction(features, prediction, confidence, latency_ms, version)
            PREDICTIONS.labels(version=str(version), class_name=CLASS_NAMES[prediction]).inc()
            LATENCY.observe(elapsed)
        return {"prediction_id": prediction_id, "prediction": prediction,
                "class_name": CLASS_NAMES[prediction], "confidence": round(confidence, 5),
                "fraud_probability": round(fraud_probability, 5),
                "model_version": version, "latency_ms": round(latency_ms, 3)}

    def start_monitor(self):
        if not self.settings.auto_monitor or self._worker is not None:
            return
        self._stop.clear()
        self._worker = Thread(target=self._loop, daemon=True, name="helios-monitor")
        self._worker.start()

    def _loop(self):
        while not self._stop.wait(self.settings.monitor_interval_seconds):
            try:
                result = self.recovery.run_cycle()
                LOG.info("HELIOS monitoring cycle: %s", result["state"])
            except Exception:
                LOG.exception("Monitoring cycle failed; will retry on next interval")

    def stop_monitor(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2)
            self._worker = None
