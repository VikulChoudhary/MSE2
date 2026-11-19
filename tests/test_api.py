from helios.training import FEATURE_NAMES

AUTH = {"X-Admin-Key": "unit-test-secret"}
NORMAL = {"amount":65, "hour":12, "merchant_risk":0.15, "international":0, "transactions_24h":2, "account_age_days":650}


def test_bootstrap_predict_and_dashboard(app_and_client):
    _, client = app_and_client
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "healthy"
    assert client.get("/").status_code == 200
    assert "HELIOS" in client.get("/").text
    response = client.post("/predict", json=NORMAL)
    assert response.status_code == 200
    result = response.json()
    assert result["class_name"] == "legitimate"
    assert result["prediction"] == 0 and result["model_version"] == 1
    assert 0 <= result["confidence"] <= 1
    snapshot = client.get("/monitor").json()
    assert snapshot["prediction_count_window"] == 1
    assert snapshot["accuracy_active_window"] is None
    assert snapshot["drift"]["ready"] is False
    assert client.get("/metrics/prometheus").status_code == 200
    assert "helios_predictions_total" in client.get("/metrics/prometheus").text


def test_rejects_invalid_input_and_protects_admin(app_and_client):
    _, client = app_and_client
    assert client.post("/predict", json={**NORMAL, "international": -1}).status_code == 422
    assert client.post("/predict", json={**NORMAL, "hour": 24}).status_code == 422
    assert client.post("/admin/recovery-cycle").status_code == 401
    assert client.post("/admin/recovery-cycle", headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.post("/feedback", json={"prediction_id": "x", "actual_class": 0}).status_code == 401
    assert client.post("/admin/recovery-cycle", headers=AUTH).json()["state"] == "healthy"


def test_feedback_is_unique_and_checked(app_and_client):
    _, client = app_and_client
    prediction = client.post("/predict", json=NORMAL).json()
    body = {"prediction_id": prediction["prediction_id"], "actual_class": prediction["prediction"]}
    assert client.post("/feedback", json=body, headers=AUTH).status_code == 201
    assert client.post("/feedback", json=body, headers=AUTH).status_code == 409
    assert client.post("/feedback", json={"prediction_id": "missing", "actual_class": 1}, headers=AUTH).status_code == 404
    assert client.get("/monitor").json()["accuracy_active_window"] == 1.0


def test_drift_detection_and_needs_labels(app_and_client):
    _, client = app_and_client
    outlier = {"amount": 8500, "hour": 2, "merchant_risk": 0.99, "international": 1, "transactions_24h": 48, "account_age_days": 1}
    for i in range(35):
        assert client.post("/predict", json=outlier).status_code == 200
    drift = client.get("/drift").json()
    assert drift["ready"] and drift["drift_detected"]
    assert set(drift["features"]) == set(FEATURE_NAMES)
    cycle = client.post("/admin/recovery-cycle", headers=AUTH).json()
    assert cycle["category"] == "data_drift"
    assert cycle["state"] == "needs_attention"
    assert cycle["action"] == "need_labeled_outcomes"
    assert client.get("/incidents").json()["incidents"][0]["category"] == "data_drift"
    again = client.post("/admin/recovery-cycle", headers=AUTH).json()
    assert again["state"] == "cooldown"


def test_demo_pipeline_failure_recovers_and_verifies(app_and_client):
    _, client = app_and_client
    injected = client.post("/admin/demo/pipeline-failure", headers=AUTH)
    assert injected.status_code == 200
    assert client.get("/health").json()["status"] == "degraded"
    result = client.post("/admin/recovery-cycle", headers=AUTH).json()
    assert result["category"] == "pipeline_failure" and result["action"] == "repair_pipeline"
    assert result["state"] == "resolved" and result["verification"]["passed"]
    assert client.get("/health").json()["status"] == "healthy"
    assert client.post("/admin/verify", headers=AUTH).json()["model"]["passed"] is True


def test_versioning_experiments_and_rollback(app_and_client):
    _, client = app_and_client
    baseline = client.get("/models").json()
    assert baseline["champion_version"] == 1
    retrain = client.post("/admin/retrain", headers=AUTH).json()
    assert retrain["accepted"] and retrain["deployed"]
    assert retrain["version"] == 2
    assert retrain["quality_gates"]["min_accuracy_passed"]
    assert len(client.get("/experiments").json()["experiments"]) == 2
    assert client.get("/models").json()["champion_version"] == 2
    result = client.post("/admin/rollback", headers=AUTH).json()
    assert result["deployment"]["active_version"] == 1
    assert result["verification"]["passed"]
    assert client.post("/predict", json=NORMAL).json()["model_version"] == 1
