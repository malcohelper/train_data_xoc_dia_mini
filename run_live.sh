#!/usr/bin/env bash
# Run analytics server and headless realtime capture together.
#
# Usage:
#   ./run_live.sh
#   ./run_live.sh --enable-clicker
#   ./run_live.sh --interval 0.8 --capture-mode roi
#
# Special flags handled here:
#   --enable-clicker   enable analytics clicker API
#   --preview          do not pass --no-preview to realtime_capture.py
#   --no-diag          do not pass --diag to realtime_capture.py

set -euo pipefail

cd "$(dirname "$0")"

DEFAULT_WEIGHTS="runs/detect/runs/detect/xocdia/weights/best.pt"
ROUNDS_DIR="rounds"
ANALYTICS_HOST="127.0.0.1"
ANALYTICS_PORT="8000"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
  else
    PY="python3.11"
  fi
fi

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PY" >&2
  echo "Set PYTHON=/path/to/python or install python3.11." >&2
  exit 1
fi

ENABLE_CLICKER=0
USE_PREVIEW=0
USE_DIAG=1
HAS_WEIGHT_OVERRIDE=0
REALTIME_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --enable-clicker)
      ENABLE_CLICKER=1
      shift
      ;;
    --preview)
      USE_PREVIEW=1
      shift
      ;;
    --no-diag)
      USE_DIAG=0
      shift
      ;;
    --weights)
      HAS_WEIGHT_OVERRIDE=1
      REALTIME_ARGS+=("$1")
      if [ $# -ge 2 ]; then
        REALTIME_ARGS+=("$2")
        shift 2
      else
        shift
      fi
      ;;
    --weights=*)
      HAS_WEIGHT_OVERRIDE=1
      REALTIME_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      sed -n '1,13p' "$0"
      exit 0
      ;;
    *)
      REALTIME_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "$HAS_WEIGHT_OVERRIDE" = "0" ] && [ ! -f "$DEFAULT_WEIGHTS" ]; then
  echo "ERROR: default weights not found: $DEFAULT_WEIGHTS" >&2
  echo "Pass an override to realtime_capture.py, e.g.:" >&2
  echo "  ./run_live.sh --weights /path/to/best.pt" >&2
  exit 1
fi

mkdir -p "$ROUNDS_DIR"

SERVER_CMD=(
  "$PY" -m analytics.serve
  --host "$ANALYTICS_HOST"
  --port "$ANALYTICS_PORT"
  --rounds-dir "$ROUNDS_DIR"
)
if [ "$ENABLE_CLICKER" = "1" ]; then
  SERVER_CMD+=(--enable-clicker)
fi

REALTIME_CMD=(
  "$PY" realtime_capture.py
  --weights "$DEFAULT_WEIGHTS"
  --rounds-dir "$ROUNDS_DIR"
  --capture-mode window
  --interval 1
)
if [ "$USE_PREVIEW" = "0" ]; then
  REALTIME_CMD+=(--no-preview)
fi
if [ "$USE_DIAG" = "1" ]; then
  REALTIME_CMD+=(--diag)
fi
if [ "${#REALTIME_ARGS[@]}" -gt 0 ]; then
  REALTIME_CMD+=("${REALTIME_ARGS[@]}")
fi

server_pid=""
cleanup() {
  if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[live] Starting analytics: ${SERVER_CMD[*]}"
"${SERVER_CMD[@]}" &
server_pid=$!

sleep 0.5
if ! kill -0 "$server_pid" >/dev/null 2>&1; then
  echo "ERROR: analytics server exited during startup." >&2
  wait "$server_pid"
  exit 1
fi

echo "[live] Analytics UI: http://${ANALYTICS_HOST}:${ANALYTICS_PORT}/frame-predict.html"
echo "[live] Starting realtime: ${REALTIME_CMD[*]}"
"${REALTIME_CMD[@]}"
