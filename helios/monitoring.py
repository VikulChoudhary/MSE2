"""Monitoring: prediction profile, PSI drift, labeled quality, latency and resources."""
import math
import os
import statistics
import psutil

from .config import Settings
from .database import Store
from .registry import Registry
from .training import FEATURE_NAMES, CLASS_NAMES


def psi_for_feature(observed: list[float], profile: dict) -> float:
    inner = profile["inner_edges"]
    counts = [0] * (len(inner) + 1)
    for value in observed:
        idx = 0
        while idx < len(inner) and float(value) >= inner[idx]:
            idx += 1
        counts[idx] += 1
    n = max(len(observed), 1)
    expected = profile["fractions"]
    eps = 1e-5
    return sum((count / n - exp) * math.log((count / n + eps) / (exp + eps))
               for count, exp in zip(counts, expected))


class Monitor:
    def __init__(self, settings: Settings, store: Store, registry: Registry):
        self.settings = settings
        self.store = store
        self.registry = registry

    def snapshot(self):
        try:
            version = self.registry.active_version()
        except RuntimeError:
            version = None
        recent = self.store.recent_predictions(self.settings.monitor_window)
        labeled = self.store.labeled(self.settings.monitor_window, version=version) if version is not None else []
        accur = (sum(row["actual"] == row["predicted"] for row in labeled) / len(labeled)) if labeled else None
        fraud_labels = [row for row in labeled if row["actual"] == 1]
        fraud_recall = (sum(row["predicted"] == 1 for row in fraud_labels)/len(fraud_labels)) if fraud_labels else None
        latency = sorted([r["latency_ms"] for r in recent])
        p95 = latency[min(len(latency) - 1, math.ceil(len(latency) * 0.95) - 1)] if latency else None
        pattern = {name: sum(r["predicted"] == j for r in recent) for j, name in enumerate(CLASS_NAMES)}
        drift = {"ready": False, "threshold": self.settings.drift_psi_threshold,
                 "features": {}, "drift_detected": False, "sample_count": len(recent)}
        if version is not None and len(recent) >= self.settings.min_drift_samples:
            metadata = self.store.version(version)
            if metadata is not None:
                scores = {}
                for j, name in enumerate(FEATURE_NAMES):
                    values = [r["features"][j] for r in recent]
                    scores[name] = round(psi_for_feature(values, metadata["reference"][name]), 5)
                drift.update({"ready": True, "features": scores,
                              "drift_detected": any(score >= self.settings.drift_psi_threshold for score in scores.values())})
        try:
            proc = psutil.Process(os.getpid())
            resources = {"rss_mb": round(proc.memory_info().rss / (1024**2), 2),
                         "cpu_percent": psutil.cpu_percent(interval=None),
                         "system_memory_percent": psutil.virtual_memory().percent}
        except psutil.Error:
            resources = {"rss_mb": None, "cpu_percent": None, "system_memory_percent": None}
        active = self.store.version(version) if version is not None else None
        return {"active_version": version, "prediction_count_window": len(recent),
                "feedback_count_active_window": len(labeled),
                "accuracy_active_window": round(accur, 5) if accur is not None else None,
                "fraud_recall_active_window": round(fraud_recall, 5) if fraud_recall is not None else None,
                "fraud_labels_active_window": len(fraud_labels),
                "latency_p95_ms": round(p95, 3) if p95 is not None else None,
                "avg_latency_ms": round(statistics.mean(latency), 3) if latency else None,
                "prediction_patterns": pattern, "drift": drift, "resources": resources,
                "validation_accuracy": active["metrics"]["accuracy"] if active else None,
                "warnings": self._warnings(len(recent), len(labeled))}

    def _warnings(self, predictions, feedback):
        warnings = []
        if predictions < self.settings.min_drift_samples:
            warnings.append(f"Drift analysis needs at least {self.settings.min_drift_samples} predictions")
        if feedback < self.settings.min_feedback_samples:
            warnings.append(f"Quality alerts need at least {self.settings.min_feedback_samples} labeled outcomes for the active model")
        return warnings
