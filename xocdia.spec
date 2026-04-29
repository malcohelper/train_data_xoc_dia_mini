# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the XocDia .app bundle.

Run via ``./build_app.sh`` (or ``pyinstaller xocdia.spec`` once PyInstaller
is installed). The spec builds a windowed (.app) bundle on macOS that
ships:

* ``app_main.py`` as the entry point (logging redirect, defaults, error dialog).
* ``best.pt`` as ``Contents/Resources/best.pt`` so the bundle is fully
  self-contained.
* ``paddle`` / ``paddleocr`` / ``ultralytics`` / ``torch`` / ``torchvision``
  collected with their compiled extensions, data files, and metadata
  (PyInstaller's ``--collect-all`` semantics) — this is what fixes the
  ``torchvision::nms does not exist`` symptom we hit with py2app, which
  failed to ship torchvision's ``.dylib`` ops next to ``torch``.

Tunables: set ``XOCDIA_WEIGHTS=/abs/path/to/best.pt`` to override the
bundled weights. ``XOCDIA_BUILD_DEBUG=1`` keeps the build in
``--debug=imports`` mode for diagnosing missing-import issues.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
)


# ---------------------------------------------------------------------------
# Paths / overrides
# ---------------------------------------------------------------------------
REPO = Path(SPECPATH).resolve()  # noqa: F821 - SPECPATH is PyInstaller global
WEIGHTS = REPO / "runs" / "detect" / "runs" / "detect" / "xocdia-2" / "weights" / "best.pt"

if os.environ.get("XOCDIA_WEIGHTS"):
    WEIGHTS = Path(os.environ["XOCDIA_WEIGHTS"]).resolve()

if not WEIGHTS.is_file():
    raise SystemExit(
        f"ERROR: trained weights not found at {WEIGHTS}\n"
        "Either train a model first (`python train.py`) or set XOCDIA_WEIGHTS."
    )


# ---------------------------------------------------------------------------
# Heavy 3rd-party packages: PyInstaller's collect_all() returns
# (datas, binaries, hiddenimports) tuples covering submodules + dylibs +
# package metadata. This is the single most important thing PyInstaller
# does better than py2app for the paddle / torch / paddleocr stack.
# ---------------------------------------------------------------------------
def _collect(name):
    try:
        return collect_all(name)
    except Exception as exc:
        print(f"[xocdia.spec] WARN: collect_all({name!r}) failed: {exc}", file=sys.stderr)
        return ([], [], [])


_torch_d, _torch_b, _torch_h = _collect("torch")
_tv_d, _tv_b, _tv_h = _collect("torchvision")
_ul_d, _ul_b, _ul_h = _collect("ultralytics")
_paddle_d, _paddle_b, _paddle_h = _collect("paddle")
_pocr_d, _pocr_b, _pocr_h = _collect("paddleocr")
_px_d, _px_b, _px_h = _collect("paddlex")
_cv_d, _cv_b, _cv_h = _collect("cv2")


# ---------------------------------------------------------------------------
# Datas: ship best.pt next to the binary as Contents/Resources/best.pt.
# PyInstaller treats the second tuple element as the destination dir
# inside the bundle. "." means the bundle root (which on macOS .app
# becomes Contents/Resources/ thanks to BUNDLE() below).
# ---------------------------------------------------------------------------
DATAS = [
    (str(WEIGHTS), "."),
] + _torch_d + _tv_d + _ul_d + _paddle_d + _pocr_d + _px_d + _cv_d


BINARIES = (
    _torch_b + _tv_b + _ul_b + _paddle_b + _pocr_b + _px_b + _cv_b
)


HIDDEN_IMPORTS = list(set(
    _torch_h + _tv_h + _ul_h + _paddle_h + _pocr_h + _px_h + _cv_h + [
        # Our own modules (PyInstaller picks them up from app_main's
        # imports, but list them defensively in case dynamic imports
        # ever sneak in).
        "realtime_capture", "pipeline", "detector", "ocr_engine",
        "ocr_postprocess", "cell_preprocessor", "classes", "window_picker",
        # pyobjc frameworks the runtime touches via runtime objc lookups
        "Quartz", "ScreenCaptureKit", "AppKit", "Foundation", "objc",
        # Tk for our error dialog
        "tkinter", "tkinter.filedialog", "tkinter.messagebox",
        # mss screen-capture fallback
        "mss", "mss.darwin",
        # Misc small deps explicitly imported in our codebase
        "yaml", "tqdm", "PIL", "numpy",
    ]
))


# ---------------------------------------------------------------------------
# Excludes: bundle size hygiene. PaddleOCR 3.x's transitive deps include
# enormous packages we never call into at inference time. These ARE safe
# to drop because:
#   * matplotlib / pandas / polars / sympy: only used by paddlex's CLI
#     and report tooling, not by ``PaddleOCR.predict``.
#   * modelscope / huggingface_hub: cloud model registries; we ship the
#     YOLO weights ourselves and PaddleOCR caches its checkpoints
#     locally on first run.
#   * pypdfium2: PDF rendering for paddlex doc pipelines; never hit by
#     image-only OCR.
# If a runtime ImportError surfaces post-build, drop the offender from
# this list.
# ---------------------------------------------------------------------------
EXCLUDES = [
    "matplotlib",
    "pandas",
    "polars",
    "sympy",
    "modelscope",
    "huggingface_hub",
    "pypdfium2", "pypdfium2_raw",
    "typer",
    "Crypto", "pycryptodome",
    "pylsd",
    "scipy.tests", "numpy.tests",
    "torch.testing", "torch.test",
    "torch.utils.tensorboard",
    "torch.distributions.testing",
    "ultralytics.engine.trainer",
    "ultralytics.hub",
    # Windows-only stdlib bits PyInstaller would otherwise warn about
    "winreg", "win32api", "win32com", "win32gui", "win32con",
    # Test suites everywhere
    "tests", "test",
]


# ---------------------------------------------------------------------------
# Analysis: PyInstaller walks app_main's import graph + the explicit
# hidden imports + collect_all results to determine what to bundle.
# ---------------------------------------------------------------------------
a = Analysis(
    ["app_main.py"],
    pathex=[str(REPO)],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)


exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="XocDia",
    debug=bool(os.environ.get("XOCDIA_BUILD_DEBUG")),
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,        # UPX corrupts torch / paddle dylibs
    console=False,    # GUI app - no Terminal window on launch
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="XocDia",
)


# Wrap the collected directory into a proper macOS .app bundle.
app = BUNDLE(
    coll,
    name="XocDia.app",
    icon=str(REPO / "icon.icns") if (REPO / "icon.icns").is_file() else None,
    bundle_identifier="com.malcohelper.xocdia",
    version="0.1.0",
    info_plist={
        "CFBundleName": "XocDia",
        "CFBundleDisplayName": "XocDia",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "LSMinimumSystemVersion": "12.3",  # ScreenCaptureKit
        "NSHumanReadableCopyright": "© malcohelper",
        # Hint shown the first time macOS prompts for Screen Recording.
        "NSCameraUsageDescription":
            "XocDia does not use the camera; this key is only present "
            "to satisfy build tooling.",
        # High-DPI-aware
        "NSHighResolutionCapable": True,
        # Keep the Dock icon visible (we use a Tk window picker on launch).
        "LSUIElement": False,
    },
)
