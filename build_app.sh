#!/usr/bin/env bash
# Build the XocDia .app bundle on macOS.
#
# Prerequisites:
#   * macOS 12.3+ (Apple Silicon recommended)
#   * Python 3.11 in the active venv
#   * The repo's normal runtime deps already installed
#     (pip install -r requirements would normally handle this; we
#      verify the critical ones below).
#   * A trained YOLO weights file at the default path
#     runs/detect/runs/detect/xocdia-2/weights/best.pt
#     (or set XOCDIA_WEIGHTS=/path/to/best.pt)
#
# Outputs:
#   * dist/XocDia.app                 – ready to drag into /Applications
#   * dist/XocDia.zip                 – ditto-compressed bundle for sharing
#
# The build is intentionally "from scratch" each time; partial caches
# under build/ have caused stale-import bugs in prior py2app revs.

set -euo pipefail

cd "$(dirname "$0")"

# --- environment sanity ------------------------------------------------------

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: build_app.sh must run on macOS (this is $(uname -s))." >&2
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "WARN: no virtualenv detected. Activate the project venv first," >&2
  echo "      otherwise py2app may bundle the system Python and fail." >&2
fi

if ! python -c "import py2app" >/dev/null 2>&1; then
  echo "Installing py2app into the active environment..."
  python -m pip install --upgrade py2app
fi

# Sanity-check the critical runtime imports before py2app tries to walk
# them. A missing import here is the #1 cause of "build succeeded but
# bundle crashes on launch" reports.
python - <<'PY'
import importlib, sys
required = [
    "cv2", "numpy", "ultralytics", "paddle", "paddleocr",
    "mss", "Quartz", "ScreenCaptureKit", "tkinter",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError as exc:
        missing.append(f"{name}: {exc}")
if missing:
    sys.exit(
        "Missing runtime imports needed to build the bundle:\n  "
        + "\n  ".join(missing)
        + "\nInstall them in the active venv before re-running build_app.sh."
    )
PY

# --- clean -------------------------------------------------------------------

echo "Cleaning previous build/ and dist/..."
rm -rf build dist

# --- py2app ------------------------------------------------------------------

echo "Building XocDia.app via py2app (this can take 5-15 minutes)..."
python setup_app.py py2app

if [[ ! -d dist/XocDia.app ]]; then
  echo "ERROR: build finished without producing dist/XocDia.app" >&2
  exit 2
fi

# --- post-build: zip for distribution ----------------------------------------

echo "Compressing dist/XocDia.app -> dist/XocDia.zip ..."
# ``ditto -c -k --keepParent`` preserves macOS xattrs/quarantine flags
# the way Finder's "Compress" command does, so recipients get a clean
# Gatekeeper experience.
ditto -c -k --keepParent dist/XocDia.app dist/XocDia.zip

SIZE_APP=$(du -sh dist/XocDia.app | awk '{print $1}')
SIZE_ZIP=$(du -sh dist/XocDia.zip | awk '{print $1}')

cat <<EOF

============================================================
  Build complete.
  Bundle:  dist/XocDia.app  ($SIZE_APP)
  Zip:     dist/XocDia.zip  ($SIZE_ZIP)

  To test locally:
      open dist/XocDia.app

  To share:
      Send dist/XocDia.zip to recipients. They unzip and follow
      USER_GUIDE.md (right-click -> Open the first time, then
      grant Screen Recording permission).
============================================================
EOF
