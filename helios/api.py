"""FastAPI routes for serving, monitoring, diagnosis and guarded operations."""
from contextlib import asynccontextmanager
from pathlib import Path
import hmac

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .config import Settings
from .service import HeliosService


class FraudRequest(BaseModel):
    amount: float = Field(ge=0, le=100000, allow_inf_nan=False)
    hour: int = Field(ge=0, le=23)
    merchant_risk: float = Field(ge=0, le=1, allow_inf_nan=False)
    international: int = Field(ge=0, le=1)
    transactions_24h: int = Field(ge=0, le=1000)
    account_age_days: int = Field(ge=0, le=50000)

    def vector(self):
        return [self.amount, self.hour, self.merchant_risk,
                self.international, self.transactions_24h, self.account_age_days]


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(min_length=1, max_length=64)
    actual_class: int = Field(ge=0, le=1)


def create_app(settings: Settings | None = None):
    settings = settings or Settings()
    service = HeliosService(settings)

    @asynccontextmanager
    async def lifespan(app):
        service.bootstrap()
        service.start_monitor()
        try:
            yield
        finally:
            service.stop_monitor()

    app = FastAPI(title="HELIOS | Autonomous Self-Healing ML Infrastructure",
                  description="Detect. Diagnose. Recover. Verify.", version=__version__, lifespan=lifespan)
    app.state.service = service

    def require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
        if not settings.admin_api_key:
            raise HTTPException(503, "Admin operations disabled. Set HELIOS_ADMIN_API_KEY.")
        if not x_admin_key or not hmac.compare_digest(x_admin_key, settings.admin_api_key):
            raise HTTPException(401, "Valid X-Admin-Key required")

    @app.get("/", include_in_schema=False)
    def homepage():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health():
        try:
            _, version = service.registry.active_model()
            pipeline = service.recovery.pipeline_check()
            if not pipeline["healthy"]:
                return {"status": "degraded", "model_version": version, "pipeline": pipeline}
            return {"status": "healthy", "model_version": version, "pipeline": pipeline}
        except Exception as exc:
            return Response(content=f'{{"status":"unhealthy","reason":"model unavailable"}}',
                            status_code=503, media_type="application/json")

    @app.post("/predict")
    def predict(data: FraudRequest):
        try:
            return service.predict(data.vector())
        except Exception as exc:
            raise HTTPException(503, "Prediction service unavailable") from exc

    @app.post("/feedback", status_code=201, dependencies=[Depends(require_admin)])
    def feedback(data: FeedbackRequest):
        try:
            service.store.add_feedback(data.prediction_id, data.actual_class)
            return {"status": "recorded", "prediction_id": data.prediction_id}
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/monitor")
    def monitor():
        return service.monitor.snapshot()

    @app.get("/drift")
    def drift():
        return service.monitor.snapshot()["drift"]

    @app.get("/models")
    def models():
        try:
            return service.registry.models()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/experiments")
    def experiments(limit: int = 25):
        return {"experiments": service.store.experiments(min(max(limit, 1), 100))}

    @app.get("/incidents")
    def incidents(limit: int = 25):
        return {"incidents": service.store.incidents(min(max(limit, 1), 100))}

    @app.get("/metrics/prometheus")
    def prometheus():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/recovery-cycle", dependencies=[Depends(require_admin)])
    def recovery_cycle():
        return service.recovery.run_cycle()

    @app.post("/admin/retrain", dependencies=[Depends(require_admin)])
    def retrain():
        return service.trainer.train(source="manual:api")

    @app.post("/admin/rollback", dependencies=[Depends(require_admin)])
    def rollback():
        try:
            deployment = service.registry.rollback()
            verification = service.recovery._verify_model()
            return {"deployment": deployment, "verification": verification}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/admin/verify", dependencies=[Depends(require_admin)])
    def verify():
        return {"model": service.recovery._verify_model(),
                "pipeline": service.recovery.pipeline_check(), "monitor": service.monitor.snapshot()}

    @app.post("/admin/demo/pipeline-failure", dependencies=[Depends(require_admin)])
    def simulate_pipeline_failure():
        if not settings.demo_mode:
            raise HTTPException(403, "Demo fault injection requires HELIOS_DEMO_MODE=true")
        service.recovery.fault_file.write_text("simulated pipeline failure", encoding="utf-8")
        return {"status": "injected", "next_step": "POST /admin/recovery-cycle"}

    return app
