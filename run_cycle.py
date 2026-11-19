"""Perform one diagnosis and recovery attempt (for cron, etc.)."""
import json
from helios.config import Settings
from helios.service import HeliosService

if __name__ == "__main__":
    service = HeliosService(Settings())
    service.bootstrap()
    print(json.dumps(service.recovery.run_cycle(), indent=2))
