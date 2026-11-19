"""Start with: uvicorn main:app --host 0.0.0.0 --port 8000"""
from helios.api import create_app

app = create_app()
