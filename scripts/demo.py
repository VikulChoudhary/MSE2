"""Self-contained end-to-end demonstration. Run: python -m scripts.demo"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from fastapi.testclient import TestClient
from helios.api import create_app
from helios.config import Settings


def main():
    with TemporaryDirectory(prefix="helios-demo-") as tmp:
        settings = replace(Settings(), data_dir=Path(tmp), admin_api_key="demo-secret",
                           auto_monitor=False, demo_mode=True, action_cooldown_seconds=0)
        app = create_app(settings)
        with TestClient(app) as client:
            auth = {"X-Admin-Key": "demo-secret"}
            print("1. Bootstrapped champion:", client.get("/models").json()["champion_version"])
            normal = {"amount": 65, "hour": 12, "merchant_risk": 0.15,
                      "international": 0, "transactions_24h": 2, "account_age_days": 650}
            print("2. Prediction:", json.dumps(client.post("/predict", json=normal).json()))
            shifted = {"amount": 8500, "hour": 2, "merchant_risk": 0.99,
                       "international": 1, "transactions_24h": 48, "account_age_days": 1}
            for _ in range(35):
                client.post("/predict", json=shifted)
            print("3. Drift alarm:", client.get("/drift").json()["drift_detected"])
            result = client.post("/admin/recovery-cycle", headers=auth).json()
            print("4. Diagnosis:", result["category"], "| Recovery:", result["action"], "| State:", result["state"])
            print("   This is a SAFE refusal to retrain without reliable labels.")
            client.post("/admin/demo/pipeline-failure", headers=auth)
            result = client.post("/admin/recovery-cycle", headers=auth).json()
            print("5. Pipeline fault:", result["category"], "| Recovery:", result["action"], "| Verified:", result["verification"]["passed"])
            trained = client.post("/admin/retrain", headers=auth).json()
            print("6. Versioned retrain:", trained["version"], "| Accepted:", trained["accepted"])
            rolled = client.post("/admin/rollback", headers=auth).json()
            print("7. Rollback:", rolled["deployment"]["active_version"], "| Verified:", rolled["verification"]["passed"])
            print("8. Incidents tracked:", len(client.get("/incidents").json()["incidents"]))
            print("Demo complete. Data was isolated in a temporary directory.")


if __name__ == "__main__":
    main()
