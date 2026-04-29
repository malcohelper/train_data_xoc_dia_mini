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


# ---------------------------------------------------------------------------
# Patch ``imp.find_module`` so it understands PEP 420 namespace packages.
#
# py2app's ``detect_dunder_file`` recipe collects every package that
# carries a ``__file__`` reference in the modulegraph, then calls
# ``imp.find_module(pkg)`` to locate its bootstrap. ``ruamel`` (used
# transitively by paddleocr / paddlex) ships only as ``ruamel.yaml`` -
# the bare ``ruamel`` namespace has no ``__init__.py``, so the legacy
# ``imp.find_module`` raises ``ImportError`` and the whole build dies.
#
# The fallback below uses ``importlib`` (PEP 451) to resolve namespace
# packages and re-shapes the result into the (file, pathname, description)
# triple ``imp.find_module`` callers expect.
# ---------------------------------------------------------------------------
import imp as _imp
import importlib.util as _ilu

_original_find_module = _imp.find_module


def _find_module_with_namespace_fallback(name, path=None):
    try:
        return _original_find_module(name, path)
    except ImportError:
        # PEP 420 namespace package handling.
        spec = _ilu.find_spec(name) if path is None else None
        if spec is None or spec.submodule_search_locations is None:
            raise
        ns_path = list(spec.submodule_search_locations)[0]
        return (None, ns_path, ("", "", _imp.PKG_DIRECTORY))


_imp.find_module = _find_module_with_namespace_fallback


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

# Files / dirs we never want shipped. PaddleOCR 3.x drags in a
# *huge* dependency tree (modelscope, pandas, polars, sympy,
# matplotlib, pypdfium2, …) intended for cloud serving. We only need
# the core OCR detector + recogniser, so we exclude the heavyweights
# and rely on PaddleOCR's lazy imports to never reach them at runtime.
# If a runtime ImportError surfaces post-build, move the offending
# package back from EXCLUDES to PACKAGES.
EXCLUDES = [
    "tests", "tools",
    "tkinter.test",

    # ---- paddle / paddleocr internals we don't use at inference ----
    "paddle.dataset", "paddle.fluid.tests", "paddle.tests",
    "paddle.utils.cpp_extension",
    "paddleocr.tools", "paddleocr.deploy",

    # ---- torch test / dev internals ----
    "torch.testing", "torch.test",
    "torch.utils.tensorboard",

    # ---- ultralytics dev/training utilities ----
    "ultralytics.engine.trainer", "ultralytics.hub",
    "ultralytics.data.scripts", "ultralytics.utils.benchmarks",

    # ---- huge transitive deps PaddleOCR 3.x pulls in but our
    #      inference path never imports. If anything breaks at
    #      runtime, move the offender back into PACKAGES. ----
    "modelscope",         # cloud model registry (~hundreds of MB)
    "pypdfium2", "pypdfium2_raw",  # PDF rendering (PaddleX feature)
    "matplotlib",         # plotting; unused at inference
    "polars",             # dataframe; only used by some PaddleX tools
    "pandas",             # ditto - inference path doesn't touch it
    "sympy",              # symbolic math; used only by torch.fx
    "scipy",              # statistics; not in the YOLO inference path
    "huggingface_hub",    # model download from HF; we cache locally
    "typer",              # CLI framework; only used by paddlex CLI
    "Crypto", "pycryptodome",  # paddleocr CLI signing; unused
    "pylsd",              # line-segment detection; doc unwarping
    "shapely.tests",
    "PIL.tests",

    # ---- ruamel namespace package: py2app's detect_dunder_file
    # recipe can't bootstrap the bare ``ruamel`` namespace (no
    # __init__.py). ruamel.yaml is only used by paddlex config
    # loaders we don't hit at inference time, so exclude entirely.
    "ruamel", "ruamel.yaml",

    # ---- Windows-only / dev-only Python stdlib bits ----
    "_winreg", "winreg",
    "win32api", "win32com", "win32con", "win32gui",
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
