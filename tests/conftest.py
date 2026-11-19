import pytest
from fastapi.testclient import TestClient
from helios.api import create_app
from helios.config import Settings


@pytest.fixture
def app_and_client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", admin_api_key="unit-test-secret",
                        auto_monitor=False, action_cooldown_seconds=0, demo_mode=True)
    app = create_app(settings)
    with TestClient(app) as client:
        yield app, client
