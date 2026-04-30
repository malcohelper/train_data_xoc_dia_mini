#!/usr/bin/env bash
# All-in-one launcher for the XocDia source distribution on macOS.
#
# Usage:
#   chmod +x run.sh
#   ./run.sh                   # interactive: pick weights if multiple
#   ./run.sh --weights path    # explicit weights, skip the picker
#   ./run.sh --no-diag         # forwarded to realtime_capture.py
#
# What it does:
#   1) Locates a Python 3.11 interpreter (or aborts with install hint).
#   2) Creates ./venv if it doesn't exist.
#   3) Installs requirements.txt the first time (cached via a marker
#      file so subsequent launches are instant).
#   4) Discovers YOLO weights under runs/detect/**/best.pt and either
#      auto-picks (if exactly one) or prompts a numbered menu.
#   5) Execs realtime_capture.py with --diag and the chosen weights.
#
# Compatible with macOS' default /bin/bash 3.2 (no `mapfile`, no
# associative arrays, no `${arr[*]@Q}`).
set -euo pipefail

cd "$(dirname "$0")"

# ---------- 1. find python ----------
PY=""
for candidate in python3.11 python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
    case "$ver" in
      3.11|3.12)
        PY="$candidate"
        break
        ;;
    esac
  fi
done

if [ -z "$PY" ]; then
  cat >&2 <<'MSG'
ERROR: Python 3.11 (or 3.12) was not found on PATH.

Install Python 3.11 from one of:
  * https://www.python.org/downloads/macos/
  * brew install python@3.11

Then re-run this script.
MSG
  exit 1
fi
echo "[run] Using $PY ($($PY --version))"

# ---------- 1b. tkinter sanity check ----------
if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
  cat >&2 <<MSG
ERROR: '$PY' was built without tkinter (the window picker dialog
needs it). Fix it with one of:
  * brew install python-tk@3.11
  * Reinstall Python from https://www.python.org/downloads/macos/
    (the python.org installer bundles Tk).
MSG
  exit 1
fi

# ---------- 2. venv ----------
if [ ! -d venv ]; then
  echo "[run] Creating virtualenv ./venv (one-time, ~30s)..."
  "$PY" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
PY=python  # the venv's python from here on

# ---------- 3. dependencies ----------
DEPS_MARKER="venv/.deps_installed"
if [ ! -f "$DEPS_MARKER" ] || [ requirements.txt -nt "$DEPS_MARKER" ]; then
  echo "[run] Installing dependencies (first run, ~5-10 min)..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
  date > "$DEPS_MARKER"
  echo "[run] Dependencies installed."
fi

# ---------- 4. weights picker ----------
WEIGHTS_OVERRIDE=""
PASSTHRU=()
while [ $# -gt 0 ]; do
  case "$1" in
    --weights)
      WEIGHTS_OVERRIDE="$2"
      shift 2
      ;;
    --weights=*)
      WEIGHTS_OVERRIDE="${1#--weights=}"
      shift
      ;;
    *)
      PASSTHRU+=("$1")
      shift
      ;;
  esac
done

if [ -n "$WEIGHTS_OVERRIDE" ]; then
  PICK="$WEIGHTS_OVERRIDE"
  if [ ! -f "$PICK" ]; then
    echo "ERROR: --weights file not found: $PICK" >&2
    exit 1
  fi
  echo "[run] Using weights from --weights: $PICK"
else
  # Collect best.pt files under runs/detect (any depth).
  WEIGHTS_LIST=()
  while IFS= read -r line; do
    WEIGHTS_LIST+=("$line")
  done < <(find runs/detect -type f -name best.pt 2>/dev/null | sort)

  if [ "${#WEIGHTS_LIST[@]}" -eq 0 ]; then
    cat >&2 <<'MSG'
ERROR: No best.pt found under runs/detect/.

Expected layout:
  runs/detect/<run-name>/weights/best.pt
  runs/detect/runs/detect/<run-name>/weights/best.pt   (nested ok)

Either train a model first (see README.md) or pass an explicit
weights file:
  ./run.sh --weights /path/to/best.pt
MSG
    exit 1
  elif [ "${#WEIGHTS_LIST[@]}" -eq 1 ]; then
    PICK="${WEIGHTS_LIST[0]}"
    echo "[run] Auto-picked the only weights available:"
    echo "      $PICK"
  else
    echo "[run] Multiple weights available, choose one:"
    i=1
    for w in "${WEIGHTS_LIST[@]}"; do
      printf "  [%d] %s\n" "$i" "$w"
      i=$((i + 1))
    done
    while :; do
      printf "Pick a number (1-%d): " "${#WEIGHTS_LIST[@]}"
      read -r choice
      case "$choice" in
        ''|*[!0-9]*)
          echo "  -> not a number, try again"
          ;;
        *)
          if [ "$choice" -ge 1 ] && [ "$choice" -le "${#WEIGHTS_LIST[@]}" ]; then
            PICK="${WEIGHTS_LIST[$((choice - 1))]}"
            break
          fi
          echo "  -> out of range, try again"
          ;;
      esac
    done
    echo "[run] Using weights: $PICK"
  fi
fi

# ---------- 5. launch ----------
# --diag prints per-frame detection counts which is invaluable when
# things go sideways. Users can disable it via --no-diag (which we
# accept as a passthrough).
USE_DIAG=1
FILTERED=()
for arg in "${PASSTHRU[@]+"${PASSTHRU[@]}"}"; do
  if [ "$arg" = "--no-diag" ]; then
    USE_DIAG=0
  else
    FILTERED+=("$arg")
  fi
done

CMD=("$PY" realtime_capture.py --weights "$PICK")
if [ "$USE_DIAG" = "1" ]; then
  CMD+=(--diag)
fi
if [ "${#FILTERED[@]}" -gt 0 ]; then
  CMD+=("${FILTERED[@]}")
fi

echo "[run] Launching: ${CMD[*]}"
exec "${CMD[@]}"
