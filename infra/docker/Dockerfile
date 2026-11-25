FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md requirements.lock /app/
COPY recsys /app/recsys
COPY scripts /app/scripts
COPY docs/benchmarks /app/docs/benchmarks
COPY notices /app/notices
RUN python -m pip install --no-cache-dir -r requirements.lock && python -m pip install --no-deps .
RUN python scripts/build_demo.py --output-dir /app/artifacts/demo

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/artifacts/demo /app/artifacts/demo
COPY notices /app/notices
RUN useradd --create-home appuser
USER appuser
ENV RECSYS_ARTIFACT_DIR=/app/artifacts/demo PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/readyz', timeout=3)"
CMD ["python", "-m", "uvicorn", "recsys.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
