#!/bin/bash
# Start the Cognifold HTTP service.
#
# Configuration via environment variables:
#   COGNIFOLD_HOST            Bind host (default: 127.0.0.1)
#   COGNIFOLD_PORT            Bind port (default: 8000)
#   COGNIFOLD_LOG_LEVEL       Log level: debug|info|warning|error (default: info)
#   COGNIFOLD_PERSIST_DIR     Session persistence directory (default: ./sessions)
#   COGNIFOLD_API_KEY         API key for authentication (default: none)
#   COGNIFOLD_SESSION_BACKEND Session store: file|redis (default: file)
#   COGNIFOLD_REDIS_URL       Redis URL (default: redis://localhost:6379/0)
#   COGNIFOLD_GUNICORN        Set to "1" to use Gunicorn (default: uvicorn)
#   COGNIFOLD_WORKERS         Worker count (default: 1)
#
# If cognifold is not installed on PATH, set PYTHONPATH=src before running:
#   PYTHONPATH=src ./scripts/start_server.sh

set -e

HOST="${COGNIFOLD_HOST:-127.0.0.1}"
PORT="${COGNIFOLD_PORT:-8000}"
LOG_LEVEL="${COGNIFOLD_LOG_LEVEL:-info}"
PERSIST_DIR="${COGNIFOLD_PERSIST_DIR:-./sessions}"
API_KEY="${COGNIFOLD_API_KEY:-}"
SESSION_BACKEND="${COGNIFOLD_SESSION_BACKEND:-file}"
REDIS_URL="${COGNIFOLD_REDIS_URL:-redis://localhost:6379/0}"
WORKERS="${COGNIFOLD_WORKERS:-1}"
USE_GUNICORN="${COGNIFOLD_GUNICORN:-0}"

# Build command arguments
ARGS=(
    serve
    --host "$HOST"
    --port "$PORT"
    --log-level "$LOG_LEVEL"
    --persist-dir "$PERSIST_DIR"
    --session-backend "$SESSION_BACKEND"
    --redis-url "$REDIS_URL"
    --workers "$WORKERS"
)

if [ -n "$API_KEY" ]; then
    ARGS+=(--api-key "$API_KEY")
    AUTH_STATUS="enabled (key configured)"
else
    ARGS+=(--no-auth)
    AUTH_STATUS="disabled"
fi

if [ "$USE_GUNICORN" = "1" ]; then
    ARGS+=(--gunicorn)
    RUNNER="gunicorn"
else
    RUNNER="uvicorn"
fi

echo "========================================"
echo "  Cognifold Service"
echo "========================================"
echo "  URL:      http://${HOST}:${PORT}"
echo "  Docs:     http://${HOST}:${PORT}/docs"
echo "  Auth:     ${AUTH_STATUS}"
echo "  Log:      ${LOG_LEVEL}"
echo "  Data:     ${PERSIST_DIR}"
echo "  Backend:  ${SESSION_BACKEND}"
echo "  Runner:   ${RUNNER} (${WORKERS} workers)"
echo "========================================"
echo ""

exec python3 -m cognifold "${ARGS[@]}"
