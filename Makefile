.PHONY: test train api docker-up

test:
	python -m unittest discover -s tests

train:
	python scripts/train_local.py --output-dir artifacts/local

api:
	uvicorn recsys.serving.api:app --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build
