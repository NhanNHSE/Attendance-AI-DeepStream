#!/bin/bash
set -euo pipefail

# Use IF_REST_* environment variables (matching docker-compose) with sensible defaults
: "${IF_REST_HTTP_PORT:=18081}"
: "${IF_REST_WORKERS:=1}"
: "${LOG_LEVEL:=info}"

echo "Preparing models..."
python -m if_rest.prepare_models

echo "Starting InsightFace-REST using ${IF_REST_WORKERS} workers on port ${IF_REST_HTTP_PORT}."

exec gunicorn --log-level "${LOG_LEVEL}" \
     -w "${IF_REST_WORKERS}" \
     -k uvicorn.workers.UvicornWorker \
     --keep-alive 60 \
     --timeout 60 \
     if_rest.api.main:app -b 0.0.0.0:${IF_REST_HTTP_PORT}