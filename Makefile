.PHONY: test lint train tune build-demo demo api docker-up

test:
	python -m unittest discover -s tests

lint:
	ruff check recsys scripts tests

build-demo:
	python scripts/build_demo.py

train:
	python scripts/train_local.py --output-dir artifacts/local

tune:
	python scripts/tune_local.py

demo:
	RECSYS_ARTIFACT_DIR=artifacts/demo uvicorn recsys.serving.api:app --host 127.0.0.1 --port 8000

api:
	uvicorn recsys.serving.api:app --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build
