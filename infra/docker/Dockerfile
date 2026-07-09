FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY recsys /app/recsys
COPY configs /app/configs
COPY scripts /app/scripts
RUN python -m pip install --no-cache-dir -e .
EXPOSE 8000
