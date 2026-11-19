"""Settings are read from environment variables at application creation time."""
from dataclasses import dataclass, field
from pathlib import Path
import os


def env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in {"true", "yes", "1", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("HELIOS_DATA_DIR", "runtime")).resolve())
    admin_api_key: str = field(default_factory=lambda: os.getenv("HELIOS_ADMIN_API_KEY", ""))
    auto_monitor: bool = field(default_factory=lambda: env_bool("HELIOS_AUTO_MONITOR", True))
    monitor_interval_seconds: int = field(default_factory=lambda: int(os.getenv("HELIOS_MONITOR_INTERVAL_SECONDS", "60")))
    monitor_window: int = field(default_factory=lambda: int(os.getenv("HELIOS_MONITOR_WINDOW", "100")))
    min_drift_samples: int = field(default_factory=lambda: int(os.getenv("HELIOS_MIN_DRIFT_SAMPLES", "30")))
    drift_psi_threshold: float = field(default_factory=lambda: float(os.getenv("HELIOS_DRIFT_PSI_THRESHOLD", "0.25")))
    min_feedback_samples: int = field(default_factory=lambda: int(os.getenv("HELIOS_MIN_FEEDBACK_SAMPLES", "20")))
    accuracy_alert_threshold: float = field(default_factory=lambda: float(os.getenv("HELIOS_ACCURACY_ALERT_THRESHOLD", "0.85")))
    candidate_accuracy_threshold: float = field(default_factory=lambda: float(os.getenv("HELIOS_CANDIDATE_ACCURACY_THRESHOLD", "0.90")))
    min_fraud_recall: float = field(default_factory=lambda: float(os.getenv("HELIOS_MIN_FRAUD_RECALL", "0.65")))
    max_allowed_regression: float = field(default_factory=lambda: float(os.getenv("HELIOS_MAX_ALLOWED_REGRESSION", "0.03")))
    latency_alert_ms: float = field(default_factory=lambda: float(os.getenv("HELIOS_LATENCY_ALERT_MS", "500")))
    action_cooldown_seconds: int = field(default_factory=lambda: int(os.getenv("HELIOS_ACTION_COOLDOWN_SECONDS", "300")))
    demo_mode: bool = field(default_factory=lambda: env_bool("HELIOS_DEMO_MODE", False))
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("HELIOS_MLFLOW_TRACKING_URI", ""))

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "helios.sqlite3"

    def validate(self) -> None:
        if self.monitor_interval_seconds < 1 or self.action_cooldown_seconds < 0:
            raise ValueError("Monitoring interval must be positive, and cooldown nonnegative")
        if min(self.monitor_window, self.min_drift_samples, self.min_feedback_samples) < 1:
            raise ValueError("Monitor window and minimum sample counts must be positive")
        if not 0 <= self.accuracy_alert_threshold <= 1 or not 0 <= self.candidate_accuracy_threshold <= 1:
            raise ValueError("Accuracy thresholds must be in [0, 1]")
        if not 0 <= self.min_fraud_recall <= 1:
            raise ValueError("Fraud recall threshold must be in [0, 1]")
        if self.max_allowed_regression < 0:
            raise ValueError("Allowed regression must be nonnegative")
