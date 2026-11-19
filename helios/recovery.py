"""Rule-based root cause analysis, safe recovery and post-action verification.

This is a transparent deterministic policy engine, not an LLM that can execute
arbitrary commands. Recovery actions are restricted to vetted operations.
"""
from datetime import datetime, timezone
from threading import RLock
from time import monotonic


from .config import Settings
from .database import Store
from .monitoring import Monitor
from .registry import Registry
from .training import Trainer, dataset


class RecoveryEngine:
    def __init__(self, settings: Settings, store: Store, registry: Registry, monitor: Monitor, trainer: Trainer):
        self.settings, self.store, self.registry = settings, store, registry
        self.monitor, self.trainer = monitor, trainer
        self.lock = RLock()
        self.fault_file = settings.data_dir / "pipeline_failure.flag"

    def pipeline_check(self):
        if self.fault_file.exists():
            return {"healthy": False, "reason": "Simulated input pipeline failure"}
        try:
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
            self.settings.model_dir.mkdir(parents=True, exist_ok=True)
            with self.store.connect() as db:
                db.execute("SELECT 1").fetchone()
            train_x, _, _, _ = dataset()
            if not train_x or len(train_x[0]) != 6:
                raise RuntimeError("Fraud feature schema changed")
            return {"healthy": True, "reason": "Dataset, local storage and schema available"}
        except Exception as exc:
            return {"healthy": False, "reason": str(exc)}

    def repair_pipeline(self):
        # Limited, auditable repair: clear demo fault and recreate expected local directories.
        self.settings.model_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.demo_mode:
            self.fault_file.unlink(missing_ok=True)
        return self.pipeline_check()

    def diagnose(self, snapshot: dict):
        pipeline = self.pipeline_check()
        if not pipeline["healthy"]:
            return "pipeline_failure", "Input pipeline/storage health check failed", {"pipeline": pipeline}
        try:
            model, _ = self.registry.active_model()
            model.predict([[65.0, 12, 0.15, 0, 2, 650]])
        except Exception as exc:
            return "model_unavailable", "Active model cannot be loaded or queried", {"exception": str(exc)}
        sufficient_feedback = snapshot["feedback_count_active_window"] >= self.settings.min_feedback_samples
        if sufficient_feedback and snapshot["accuracy_active_window"] < self.settings.accuracy_alert_threshold:
            return "model_degradation", "Labeled prediction accuracy is below alert threshold", {
                "accuracy": snapshot["accuracy_active_window"], "threshold": self.settings.accuracy_alert_threshold,
                "drift_detected": snapshot["drift"]["drift_detected"],
                "feedback_count": snapshot["feedback_count_active_window"]}
        if snapshot["drift"]["ready"] and snapshot["drift"]["drift_detected"]:
            return "data_drift", "Recent feature distribution differs from training baseline (PSI)", {
                "psi": snapshot["drift"]["features"], "threshold": self.settings.drift_psi_threshold,
                "feedback_count": snapshot["feedback_count_active_window"]}
        if (snapshot["prediction_count_window"] >= self.settings.min_drift_samples and
                snapshot["latency_p95_ms"] is not None and
                snapshot["latency_p95_ms"] > self.settings.latency_alert_ms):
            return "latency_degradation", "Observed p95 request latency exceeds policy", {
                "p95_ms": snapshot["latency_p95_ms"], "threshold_ms": self.settings.latency_alert_ms}
        return None

    def _verify_model(self):
        try:
            score = self.trainer.evaluate_version()
            good = (score["accuracy"] >= self.settings.candidate_accuracy_threshold and
                    score["fraud_recall"] >= self.settings.min_fraud_recall)
            return {"passed": good, "holdout_evaluation": score, "check": "active_model_fixed_holdout"}
        except Exception as exc:
            return {"passed": False, "check": "active_model_fixed_holdout", "error": str(exc)}

    def _select_action(self, category: str, evidence: dict, snapshot: dict):
        if category == "pipeline_failure":
            return "repair_pipeline"
        if category == "model_unavailable":
            return "rollback"
        if category == "model_degradation":
            if evidence.get("drift_detected"):
                return "retrain"
            return "rollback" if self.registry.store.previous_deployed_versions(snapshot["active_version"]) else "retrain"
        if category == "data_drift":
            return "retrain" if evidence["feedback_count"] >= self.settings.min_feedback_samples else "need_labeled_outcomes"
        if category == "latency_degradation":
            return "rollback" if self.registry.store.previous_deployed_versions(snapshot["active_version"]) else "manual_review"
        return "manual_review"

    def _cooling_down(self, incident: dict | None, action: str, snapshot: dict):
        if not incident or not incident["action_at"]:
            return False
        previous_action = incident["action"]
        if previous_action == "need_labeled_outcomes":
            return action == "need_labeled_outcomes"
        if previous_action == "manual_review":
            return action == "manual_review"
        # If the active version and available feedback haven't changed, avoid blind retraining.
        previous = incident.get("verification") or {}
        if (previous.get("active_version") == snapshot["active_version"] and
                previous.get("feedback_count") == snapshot["feedback_count_active_window"] and
                action == "retrain"):
            return True
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(incident["action_at"])).total_seconds()
        return elapsed < self.settings.action_cooldown_seconds

    def run_cycle(self):
        with self.lock:
            before = self.monitor.snapshot()
            diagnosis = self.diagnose(before)
            if diagnosis is None:
                self.store.resolve_incidents({"pipeline_failure", "model_unavailable", "model_degradation",
                                              "data_drift", "latency_degradation"},
                                             {"passed": True, "reason": "Healthy monitoring cycle"})
                return {"state": "healthy", "action": "none", "snapshot": before}
            category, cause, evidence = diagnosis
            action = self._select_action(category, evidence, before)
            existing = self.store.find_unresolved_incident(category)
            incident_id = self.store.create_or_update_incident(category, cause, evidence)
            if self._cooling_down(existing, action, before):
                return {"state": "cooldown", "category": category, "action": "none",
                        "incident_id": incident_id, "reason": "Awaiting new evidence or action cooldown"}
            if action in {"need_labeled_outcomes", "manual_review"}:
                verification = {"passed": False, "reason": action,
                                "feedback_count": before["feedback_count_active_window"],
                                "active_version": before["active_version"]}
                self.store.update_incident(incident_id, "needs_attention", action, verification)
                return {"state": "needs_attention", "category": category, "action": action,
                        "incident_id": incident_id, "verification": verification}
            try:
                if action == "repair_pipeline":
                    check = self.repair_pipeline()
                    verification = {"passed": check["healthy"], "pipeline": check}
                elif action == "rollback":
                    deployment = self.registry.rollback()
                    verification = self._verify_model()
                    verification["deployment"] = deployment
                elif action == "retrain":
                    trained = self.trainer.train(source=f"recovery:{category}")
                    verification = self._verify_model()
                    verification["training"] = trained
                    verification["passed"] = bool(trained["deployed"] and verification["passed"])
                else:
                    raise RuntimeError(f"Unsupported recovery action: {action}")
                if action == "retrain" and category == "data_drift" and verification["passed"]:
                    # Candidate holdout quality passing does not prove live drift has disappeared.
                    new_drift = self.monitor.snapshot()["drift"]
                    verification["post_recovery_drift"] = new_drift
                    if new_drift["drift_detected"]:
                        verification["passed"] = False
                        verification["reason"] = "Holdout passed, but live drift remains; continue monitoring"
                after = self.monitor.snapshot()
                verification["active_version"] = after["active_version"]
                verification["feedback_count"] = after["feedback_count_active_window"]
                state = "resolved" if verification["passed"] else "needs_attention"
                self.store.update_incident(incident_id, state, action, verification)
                return {"state": state, "category": category, "action": action,
                        "incident_id": incident_id, "verification": verification}
            except Exception as exc:
                verification = {"passed": False, "error": str(exc),
                                "active_version": before["active_version"],
                                "feedback_count": before["feedback_count_active_window"]}
                self.store.update_incident(incident_id, "needs_attention", action, verification)
                return {"state": "needs_attention", "category": category, "action": action,
                        "incident_id": incident_id, "verification": verification}
