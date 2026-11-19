"""One-shot training with quality gating. Use the same storage as the API."""
import json
from helios.config import Settings
from helios.service import HeliosService

if __name__ == "__main__":
    service = HeliosService(Settings())
    service.bootstrap()
    result = service.trainer.train(source="manual:cli")
    print(json.dumps(result, indent=2))
