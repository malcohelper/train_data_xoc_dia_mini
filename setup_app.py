"""py2app build configuration for the XocDia .app bundle.

Usage (run on macOS only)::

    pip install py2app
    python setup_app.py py2app    # produces dist/XocDia.app

This is intentionally a separate file from a hypothetical ``setup.py``
so that ``python setup.py`` (used by some tooling) doesn't accidentally
trigger an app build.

Distribution flow (after build):

1. ``./build_app.sh`` to do the clean + build.
2. ``ditto -c -k --keepParent dist/XocDia.app dist/XocDia.zip``
3. Upload zip; recipients double-click ``XocDia.app`` (right-click
   → Open the first time so unsigned Gatekeeper warns once).

The end-user docs live in ``USER_GUIDE.md``; build / packaging is
documented in ``BUILD.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup

# py2app's modulegraph recurses through every imported sub-package.
# Walking paddle / paddleocr / torch with the default 1000-frame limit
# blows the stack on Python 3.11 + recent paddle. 10000 is overkill
# for safety; modulegraph itself rarely exceeds ~3000 frames in our
# tests but the headroom is free.
sys.setrecursionlimit(10000)


REPO = Path(__file__).resolve().parent
WEIGHTS = REPO / "runs" / "detect" / "runs" / "detect" / "xocdia-2" / "weights" / "best.pt"

# Allow override via env so CI / advanced users can point at a different
# trained model without editing this file.
import os
WEIGHTS_OVERRIDE = os.environ.get("XOCDIA_WEIGHTS")
if WEIGHTS_OVERRIDE:
    WEIGHTS = Path(WEIGHTS_OVERRIDE).resolve()

if not WEIGHTS.is_file():
    sys.exit(
        f"ERROR: trained weights not found at {WEIGHTS}\n"
        "Either train a model first (`python train.py`) or set the "
        "XOCDIA_WEIGHTS env var to your existing best.pt.\n"
    )

# Ship best.pt as Contents/Resources/best.pt; app_main.py picks it up.
DATA_FILES = [
    ("", [str(WEIGHTS)]),  # "" = Contents/Resources/ root
]

# Plist keys for the bundle. Screen Recording / Accessibility prompts
# from macOS use ``CFBundleDisplayName`` / ``CFBundleIdentifier`` so we
# set both to something readable.
PLIST = {
    "CFBundleName": "XocDia",
    "CFBundleDisplayName": "XocDia",
    "CFBundleIdentifier": "com.malcohelper.xocdia",
    "CFBundleShortVersionString": "0.1.0",
    "CFBundleVersion": "0.1.0",
    # NSHumanReadableCopyright shows in About panel.
    "NSHumanReadableCopyright": "© malcohelper",
    # We don't need a Dock icon when the picker dialog is up; keep
    # the default LSUIElement = False so the user can ⌘-Tab to the
    # window picker / preview window.
    "LSMinimumSystemVersion": "12.3",  # SCKit needs 12.3+
    # Hint shown in macOS prompt when first capture happens. macOS
    # attributes Screen Recording grants to the *bundle id*, so
    # signing or notarising later won't break the existing grant.
    "NSCameraUsageDescription": (
        "XocDia does not use the camera; this key is present only to "
        "satisfy build tools."
    ),
}

# Only declare top-level modules we know py2app needs to *graph*.
# Heavy 3rd-party (paddle/torch/ultralytics) are listed in PACKAGES so
# they get copied wholesale without modulegraph walking every submodule.
# Doing both (includes + packages) is what blows the recursion limit.
INCLUDES = [
    # Our modules
    "realtime_capture", "pipeline", "detector", "ocr_engine",
    "ocr_postprocess", "cell_preprocessor", "classes", "window_picker",
    # Lightweight 3rd-party imports our code touches directly.
    "cv2", "numpy", "PIL", "yaml", "tqdm",
    "tkinter", "tkinter.filedialog", "tkinter.messagebox",
    # pyobjc submodules. Listing the framework packages here (instead
    # of in PACKAGES) keeps modulegraph from descending into the
    # generated .pyi stubs.
    "Quartz", "ScreenCaptureKit", "AppKit", "Foundation", "objc",
]

# Heavy / circularly-imported packages. ``packages`` tells py2app to
# *copy the directory verbatim* without doing import-graph analysis,
# which is what saves us from RecursionError on paddle/torch.
PACKAGES = [
    "ultralytics", "torch", "torchvision",
    "paddle", "paddleocr", "paddlex",
    "mss", "shapely",
]

# Files / dirs we never want shipped. paddle's `dataset` / `tests`
# subtrees are huge and never used at inference time; same for
# torch's test suite. Excluding them shaves hundreds of MB and cuts
# the modulegraph traversal that triggered the original RecursionError.
EXCLUDES = [
    "tests", "tools",
    "tkinter.test",
    # paddle / paddleocr internals we don't need at runtime
    "paddle.dataset", "paddle.fluid.tests", "paddle.tests",
    "paddle.utils.cpp_extension",
    "paddleocr.tools", "paddleocr.deploy",
    # torch test data
    "torch.testing", "torch.test",
    "torch.utils.tensorboard",
    # We only call ultralytics.YOLO; the trainer / hub / datasets
    # graphs pull in dozens of heavy deps (matplotlib, ray, comet, …)
    # we don't ship.
    "ultralytics.engine.trainer", "ultralytics.hub",
    "ultralytics.data.scripts", "ultralytics.utils.benchmarks",
]

OPTIONS = {
    "argv_emulation": False,  # we manage sys.argv ourselves
    "iconfile": str(REPO / "icon.icns") if (REPO / "icon.icns").is_file() else None,
    "plist": PLIST,
    "includes": INCLUDES,
    "packages": PACKAGES,
    "excludes": EXCLUDES,
    # ``site-packages`` zips break C-extensions that load resource
    # files alongside themselves (paddle is the worst offender).
    "site_packages": True,
    "semi_standalone": False,
    # Keep the bundle reasonable; we don't need .pyi stubs / tests.
    "strip": True,
    "optimize": 0,
}

setup(
    app=["app_main.py"],
    name="XocDia",
    version="0.1.0",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
