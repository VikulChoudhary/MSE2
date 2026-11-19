.PHONY: install test serve demo cycle docker
install:
	python -m pip install -r requirements.txt
test:
	python -m pytest -v
serve:
	uvicorn main:app --host 127.0.0.1 --port 8000
demo:
	python -m scripts.demo
cycle:
	python run_cycle.py
docker:
	docker compose up --build
