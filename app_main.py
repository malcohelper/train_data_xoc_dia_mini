"""Entry point used when XocDia is launched as a macOS .app bundle.

Differences vs. running ``python realtime_capture.py`` from a checkout:

* The bundle has no terminal attached, so ``stdout`` / ``stderr`` are
  redirected to ``~/Library/Logs/XocDia/<timestamp>.log`` so the user
  can read them after the fact (Console.app or ``tail -f``).
* ``best.pt`` ships inside the bundle as
  ``Contents/Resources/best.pt`` (read-only) - we point ``--weights``
  at that path automatically.
* Round JSON files have to live somewhere user-writable; we default
  to ``~/Documents/XocDia/rounds`` and create it on first run.
* PaddleOCR will lazily download its detector + recogniser checkpoints
  on first run to ``~/.paddlex``; that's the same path as the dev
  install so we don't have to override anything, but the user *does*
  need internet access for the very first launch.
* Hard errors before the realtime loop starts (missing permission,
  missing weights, model load failure) are surfaced via a
  Tkinter ``messagebox`` because there's no terminal to print to.

We also patch ``sys.argv`` so that downstream argparse defaults to
window-picker mode and the bundled weights path. Power users can
still pass extra flags by editing ``Contents/Info.plist`` →
``CFBundleArguments``, but the goal is "double-click and go".
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# ----------------------------- bundle helpers ---------------------------------


def _is_bundled() -> bool:
    """True when running inside a py2app/PyInstaller bundle.

    py2app sets ``sys.frozen = "macosx_app"``; PyInstaller sets
    ``sys.frozen = True`` and exposes ``sys._MEIPASS``. We accept both
    so the same entry point works for either packager.
    """
    return bool(getattr(sys, "frozen", False))


def _bundle_resources_dir() -> Optional[Path]:
    """Return ``Contents/Resources/`` for a py2app bundle, or ``None``
    when running from source."""
    if not _is_bundled():
        return None
    # py2app: executable is at Contents/MacOS/XocDia, resources at
    # Contents/Resources/. PyInstaller --onedir uses the same layout.
    exe = Path(sys.executable).resolve()
    candidate = exe.parent.parent / "Resources"
    return candidate if candidate.is_dir() else None


def _user_app_dir() -> Path:
    """``~/Documents/XocDia`` - user-writable home for round JSONs and
    any other artefacts the running app produces. Created on demand."""
    base = Path.home() / "Documents" / "XocDia"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _user_log_dir() -> Path:
    """``~/Library/Logs/XocDia`` - macOS-conventional location for app
    log files, visible in Console.app under "Log Reports"."""
    base = Path.home() / "Library" / "Logs" / "XocDia"
    base.mkdir(parents=True, exist_ok=True)
    return base


# --------------------------- stdout/stderr redirect ---------------------------


def _redirect_io_to_log() -> Path:
    """Send ``stdout`` and ``stderr`` to a fresh per-launch log file.

    Returns the log file path so the caller can show it to the user.
    Line-buffered so a ``tail -f`` tracks the running app live.
    """
    log_path = _user_log_dir() / f"{datetime.now():%Y%m%d_%H%M%S}.log"
    # ``buffering=1`` => line-buffered. Combined w/ ``flush=True`` in
    # the few critical prints below it's enough for tail -f.
    fp = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = fp
    sys.stderr = fp
    return log_path


# --------------------------- error dialog (no TTY) ----------------------------


def _show_error_dialog(title: str, message: str) -> None:
    """Pop a modal Tkinter messagebox so the user sees fatal errors
    even though there's no terminal attached. Falls back to a print
    when Tk isn't available (then the message at least lands in the
    log file)."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception as exc:  # noqa: BLE001
        print(f"[app] {title}: {message}  (Tk dialog failed: {exc})")


# ------------------------------- main -----------------------------------------


def _resolve_bundled_weights() -> Optional[Path]:
    res = _bundle_resources_dir()
    if res is None:
        return None
    candidate = res / "best.pt"
    return candidate if candidate.is_file() else None


def _patch_argv_for_bundle(weights: Optional[Path], rounds_dir: Path) -> None:
    """Inject sensible defaults into ``sys.argv`` *before* argparse runs.

    py2app strips the original argv anyway so this is the cleanest way
    to drive ``realtime_capture._parse_args`` without forking it.
    Users get the same defaults as ``python realtime_capture.py
    --weights <bundled> --rounds-dir <home> --capture-mode window``
    plus whatever flags they appended via Info.plist.
    """
    extra_args: list[str] = []
    if weights is not None:
        extra_args += ["--weights", str(weights)]
    extra_args += ["--rounds-dir", str(rounds_dir)]
    # Force the window picker dialog because non-technical users
    # don't have a way to drag a screen ROI from a .app launch.
    extra_args += ["--capture-mode", "window"]
    sys.argv = [sys.argv[0]] + extra_args + sys.argv[1:]


def main() -> int:
    if _is_bundled():
        log_path = _redirect_io_to_log()
        print(f"[app] log file: {log_path}")
        print(f"[app] launched at {datetime.now().isoformat(timespec='seconds')}")
    else:
        log_path = None

    # Make the working dir user-writable. mss/cv2 occasionally write
    # temp files relative to CWD; the .app bundle's CWD is /, which
    # is not writable.
    user_home = _user_app_dir()
    try:
        os.chdir(user_home)
        print(f"[app] cwd -> {user_home}")
    except OSError as exc:
        # Non-fatal: fall back to ``$HOME``. argparse paths are
        # absolute below so we don't strictly need a writable CWD.
        print(f"[app] chdir failed: {exc}; staying in {os.getcwd()}")

    weights = _resolve_bundled_weights()
    rounds_dir = user_home / "rounds"
    if _is_bundled():
        if weights is None:
            _show_error_dialog(
                "XocDia - missing model",
                "best.pt not found inside the app bundle. The build "
                "is incomplete; please re-run build_app.sh and rebuild.",
            )
            return 2
        _patch_argv_for_bundle(weights, rounds_dir)

    # Defer the heavy import so the error dialog above can fire fast
    # for the missing-weights case.
    try:
        from realtime_capture import main as rt_main
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to import realtime_capture: {exc}"
        if _is_bundled():
            _show_error_dialog("XocDia - import error", msg)
        print(msg)
        return 3

    try:
        return rt_main()
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    except KeyboardInterrupt:
        print("[app] interrupted by user")
        return 0
    except Exception as exc:  # noqa: BLE001
        # Anything that escapes ``rt_main`` is a fatal startup error
        # (model load failed, no Screen Recording permission, etc.).
        # Surface it to the user instead of silently exiting.
        msg = f"{type(exc).__name__}: {exc}"
        if log_path is not None:
            msg += f"\n\nFull traceback in:\n{log_path}"
        if _is_bundled():
            _show_error_dialog("XocDia crashed", msg)
        # Always re-raise so the traceback hits the log file.
        raise


if __name__ == "__main__":
    sys.exit(main())
