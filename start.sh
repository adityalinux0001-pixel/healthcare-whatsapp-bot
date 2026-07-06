#!/usr/bin/env bash
set -euo pipefail

WEB_CONCURRENCY="${1:-${WEB_CONCURRENCY:-4}}"
BIND="${BIND:-0.0.0.0:8000}"
TIMEOUT="${GUNICORN_TIMEOUT:-120}"
LOG_LEVEL="${LOG_LEVEL:-info}"

exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers "${WEB_CONCURRENCY}" \
    --bind "${BIND}" \
    --timeout "${TIMEOUT}" \
    --graceful-timeout 30 \
    --log-level "${LOG_LEVEL}" \
    --access-logfile - \
    --error-logfile -