from dataclasses import replace
from helios.training import Trainer


def test_rejected_candidate_never_replaces_champion(app_and_client):
    app, _ = app_and_client
    service = app.state.service
    strict_settings = replace(service.settings, candidate_accuracy_threshold=1.0)
    trainer = Trainer(strict_settings, service.store, service.registry)
    result = trainer.train(source="test:strict-gate")
    assert not result["accepted"] and not result["deployed"]
    assert result["version"] is None
    assert service.registry.active_version() == 1
    assert len(service.registry.models()["versions"]) == 1
    assert service.store.experiments()[0]["note"].startswith("Rejected")


def test_model_is_restored_after_service_restart(app_and_client):
    app, _ = app_and_client
    from helios.service import HeliosService
    copied_service = HeliosService(app.state.service.settings)
    copied_service.bootstrap()
    assert copied_service.registry.active_version() == 1
    assert copied_service.predict([65,12,0.15,0,2,650])["prediction"] == 0


def test_startup_recovers_from_missing_champion_artifact(app_and_client):
    app, client = app_and_client
    service = app.state.service
    new = client.post("/admin/retrain", headers={"X-Admin-Key": "unit-test-secret"}).json()
    assert new["version"] == 2
    # Simulate crash/corruption after deploying the second version.
    (service.settings.model_dir / "model_v2.json").unlink()
    from helios.service import HeliosService
    recovered = HeliosService(service.settings)
    recovered.bootstrap()
    assert recovered.registry.active_version() == 1
    assert recovered.trainer.evaluate_version()["accuracy"] >= 0.9
