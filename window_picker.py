"""macOS window enumeration + Tkinter dialogs + window-content capture.

Exposes the helpers used by ``realtime_capture.py``:

* ``list_windows()``                 - enumerate visible top-level windows
* ``get_window_bounds()``            - re-fetch bounds of a specific window
* ``capture_window_image()``         - grab the live pixel content of a
                                       specific window via Quartz (PR #24)
* ``screen_capture_kit_capture()``   - same idea but via ScreenCaptureKit
                                       (PR #25): works across Spaces and
                                       through fullscreen transitions
* ``pick_mode_dialog()``             - 2-button "Pick Window"/"Drag ROI"
* ``pick_window_dialog()``           - listbox to choose one window

Everything degrades gracefully on non-macOS or when ``pyobjc-framework-
Quartz`` / ``pyobjc-framework-ScreenCaptureKit`` aren't installed:
``list_windows()`` returns ``[]`` and the capture helpers return
``None`` so the caller can fall back through Quartz / mss.

PR #23 motivation (window picker)
---------------------------------
Drag-ROI lets users accidentally drag past the game window into their
own terminal, which then becomes part of the YOLO input frame and tanks
detection rates. Picking a specific window eliminates that mistake and
lets the script auto-follow the window when the user moves it.

PR #24 motivation (window-content capture via Quartz)
-----------------------------------------------------
``mss`` captures by *screen coordinates*. When any other window
(including the OpenCV preview window we render ourselves) overlaps
the game's bbox at the moment of capture, mss reads the overlay's
pixels instead of the game's. Symptom: ``dets=0`` ticks every time
the preview is repainted on top of the game window, plus a recursive
"mirror in mirror" effect when the preview *fully* covers the source.
``CGWindowListCreateImage`` reads the window's *backing store* directly
via the WindowServer, so it returns the game's pixels regardless of
occlusion or other windows on top.

PR #25 motivation (ScreenCaptureKit)
------------------------------------
``CGWindowListCreateImage`` is deprecated on macOS 14+ and, more
importantly, it does NOT capture live frames for windows in another
Space - in particular for *fullscreen* windows the user has switched
away from. Symptom in the user's log: 9 rounds detected while looking
at Safari, then ``dets=0`` for hundreds of ticks after switching to
the terminal Space (Quartz returned a stale snapshot). ``ScreenCapture
Kit`` (macOS 12.3+) captures the live backing store regardless of
which Space is foreground, so the script continues to detect even
while the user is on a different Space (e.g. reading the terminal).
We try SCKit first, then fall back to Quartz, then to mss.
"""

from __future__ import annotations

import platform
import threading
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

# Cached ScreenCaptureKit objects so we don't re-enumerate windows on
# every tick. Filled lazily on first call to
# ``screen_capture_kit_capture`` and invalidated when the window_id
# changes or when capture starts failing.
_SCK_CACHE: dict = {
    "window_id": None,
    "filter": None,
    "config": None,
}


class WindowInfo(NamedTuple):
    window_id: int
    app_name: str
    title: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) in display coords

    @property
    def label(self) -> str:
        title = self.title or "(untitled)"
        x, y, w, h = self.bbox
        return f"{self.app_name} - {title}  ({w}x{h} @ {x},{y})"


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def list_windows(min_size: int = 200) -> List[WindowInfo]:
    """Return on-screen, top-level windows large enough to be a real
    application window (default: at least 200 px on each side).

    Returns an empty list on non-macOS or when Quartz isn't available
    so callers can detect the absence and fall back to drag-ROI.
    """
    if not _is_macos():
        return []
    try:
        import Quartz  # type: ignore
    except ImportError:
        return []

    options = (
        Quartz.kCGWindowListOptionOnScreenOnly  # type: ignore[attr-defined]
        | Quartz.kCGWindowListExcludeDesktopElements  # type: ignore[attr-defined]
    )
    raw = Quartz.CGWindowListCopyWindowInfo(  # type: ignore[attr-defined]
        options, Quartz.kCGNullWindowID,  # type: ignore[attr-defined]
    )
    out: List[WindowInfo] = []
    for w in raw or []:
        # Layer 0 == normal app windows; >0 are menubar / dock / etc.
        if int(w.get("kCGWindowLayer", 0)) != 0:
            continue
        bounds = w.get("kCGWindowBounds", {}) or {}
        try:
            x = int(bounds.get("X", 0))
            y = int(bounds.get("Y", 0))
            ww = int(bounds.get("Width", 0))
            hh = int(bounds.get("Height", 0))
        except (TypeError, ValueError):
            continue
        if ww < min_size or hh < min_size:
            continue
        owner = str(w.get("kCGWindowOwnerName", "") or "")
        title = str(w.get("kCGWindowName", "") or "")
        wid = int(w.get("kCGWindowNumber", 0))
        out.append(WindowInfo(wid, owner, title, (x, y, ww, hh)))
    # Sort by area descending so the biggest (most likely the game)
    # surfaces first in the picker.
    out.sort(key=lambda wi: wi.bbox[2] * wi.bbox[3], reverse=True)
    return out


def get_window_bounds(window_id: int) -> Optional[Tuple[int, int, int, int]]:
    """Re-fetch the current bbox of a specific window. Returns ``None``
    when the window has been closed/minimised or Quartz isn't
    available."""
    if not _is_macos():
        return None
    try:
        import Quartz  # type: ignore
    except ImportError:
        return None
    info = Quartz.CGWindowListCopyWindowInfo(  # type: ignore[attr-defined]
        Quartz.kCGWindowListOptionIncludingWindow,  # type: ignore[attr-defined]
        window_id,
    )
    if not info:
        return None
    bounds = info[0].get("kCGWindowBounds", {}) or {}
    try:
        return (
            int(bounds.get("X", 0)),
            int(bounds.get("Y", 0)),
            int(bounds.get("Width", 0)),
            int(bounds.get("Height", 0)),
        )
    except (TypeError, ValueError):
        return None


def capture_window_image(
    window_id: int,
    target_size: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """Capture the live pixel content of a specific window via Quartz.

    Returns a BGR ``np.ndarray`` (compatible with ``cv2``) or ``None``
    when the window is gone, Quartz isn't installed, or we're not on
    macOS. Unlike mss screen-region capture, this reads the window's
    backing store directly through the WindowServer, so the result is
    correct regardless of:

    * other windows being on top of the target,
    * the target being minimised / on a different Space,
    * the OpenCV preview we render ourselves overlapping the game.

    ``target_size``: optional ``(width, height)`` in *logical* pixels.
    Quartz returns the image at native (physical) resolution, which is
    2x logical on Retina displays. Pass the window's logical bbox here
    to get an array the same size mss would have returned, so
    downstream coordinates stay consistent.
    """
    if not _is_macos():
        return None
    try:
        import Quartz  # type: ignore
        import cv2  # type: ignore
    except ImportError:
        return None

    try:
        cg_image = Quartz.CGWindowListCreateImage(  # type: ignore[attr-defined]
            Quartz.CGRectNull,  # type: ignore[attr-defined]
            Quartz.kCGWindowListOptionIncludingWindow,  # type: ignore[attr-defined]
            window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming  # type: ignore[attr-defined]
            | Quartz.kCGWindowImageBestResolution,  # type: ignore[attr-defined]
        )
    except Exception:  # noqa: BLE001 - surface as None, caller falls back
        return None
    if cg_image is None:
        return None

    width = int(Quartz.CGImageGetWidth(cg_image))  # type: ignore[attr-defined]
    height = int(Quartz.CGImageGetHeight(cg_image))  # type: ignore[attr-defined]
    if width <= 0 or height <= 0:
        return None
    bytes_per_row = int(
        Quartz.CGImageGetBytesPerRow(cg_image)  # type: ignore[attr-defined]
    )
    data_provider = Quartz.CGImageGetDataProvider(cg_image)  # type: ignore[attr-defined]
    raw = Quartz.CGDataProviderCopyData(data_provider)  # type: ignore[attr-defined]
    # ``CFData`` exposes a buffer protocol via pyobjc; ``np.frombuffer``
    # works directly. Each row is ``bytes_per_row`` long which may be
    # > width*4 due to alignment padding, so reshape with the row stride
    # and slice off the first ``width`` columns.
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size < bytes_per_row * height:
        return None
    arr = buf[: bytes_per_row * height].reshape((height, bytes_per_row // 4, 4))
    # Default CGImage byte order on macOS is little-endian ARGB, which
    # in memory is BGRA - so the first three channels are already BGR.
    bgr = arr[:, :width, :3].copy()  # detach from CFData backing store
    if target_size is not None and (width, height) != target_size:
        bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
    return bgr


def invalidate_sck_cache() -> None:
    """Drop cached ScreenCaptureKit filter/config. Call when the user
    picks a new window or after a stretch of capture failures so the
    next ``screen_capture_kit_capture`` rebuilds from scratch."""
    _SCK_CACHE["window_id"] = None
    _SCK_CACHE["filter"] = None
    _SCK_CACHE["config"] = None


def _sck_get_shareable_content(timeout: float = 2.0):
    """Call ``SCShareableContent.getShareableContentWithCompletionHandler:``
    synchronously by waiting on a ``threading.Event``. Returns the
    ``SCShareableContent`` instance, or ``None`` on error / timeout."""
    try:
        import ScreenCaptureKit  # type: ignore
    except ImportError:
        return None

    event = threading.Event()
    box: dict = {"content": None, "error": None}

    def handler(content, error):  # noqa: ANN001 - pyobjc bridges types
        box["content"] = content
        box["error"] = error
        event.set()

    ScreenCaptureKit.SCShareableContent.getShareableContentWithCompletionHandler_(  # type: ignore[attr-defined]
        handler,
    )
    if not event.wait(timeout):
        return None
    if box["error"] is not None:
        # macOS returns NSError when permission is denied. Surface
        # exactly once and let the caller fall back to Quartz/mss.
        print(
            f"[sckit] SCShareableContent error: {box['error']}. "
            f"If this is the first run, grant Screen Recording "
            f"permission to your terminal/Python in "
            f"System Settings -> Privacy & Security -> Screen "
            f"Recording, then re-run."
        )
        return None
    return box["content"]


def _sck_build_filter_and_config(
    window_id: int, content,
):
    """Find the SCWindow with ``window_id`` and return
    ``(SCContentFilter, SCStreamConfiguration)`` configured to capture
    that window at its native (physical) resolution. Returns
    ``(None, None)`` when the window is no longer in the shareable
    list (closed/hidden)."""
    try:
        import ScreenCaptureKit  # type: ignore
    except ImportError:
        return None, None

    target = None
    for w in content.windows() or []:
        try:
            if int(w.windowID()) == int(window_id):
                target = w
                break
        except Exception:  # noqa: BLE001
            continue
    if target is None:
        return None, None

    filter_ = (
        ScreenCaptureKit.SCContentFilter.alloc()  # type: ignore[attr-defined]
        .initWithDesktopIndependentWindow_(target)
    )
    frame = target.frame()
    width = max(1, int(frame.size.width))
    height = max(1, int(frame.size.height))

    config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()  # type: ignore[attr-defined]
    # Capture at physical resolution (2x logical on Retina); we
    # downscale to logical in the caller so the rest of the pipeline
    # stays size-consistent with the mss / Quartz paths.
    try:
        config.setWidth_(width * 2)
        config.setHeight_(height * 2)
    except AttributeError:
        # Some pyobjc versions expose properties via attribute access
        config.width = width * 2
        config.height = height * 2
    try:
        config.setShowsCursor_(False)
    except AttributeError:
        config.showsCursor = False
    # ``capturesAudio`` defaults to False; ``ignoreShadowsDisplay``
    # would clip drop-shadow framing but is a SCStream-only knob.
    return filter_, config


def screen_capture_kit_capture(
    window_id: int,
    target_size: Optional[Tuple[int, int]] = None,
    timeout: float = 1.5,
) -> Optional[np.ndarray]:
    """Capture the live pixel content of a specific window via
    ``SCScreenshotManager.captureImageWithFilter:configuration:
    completionHandler:`` (macOS 14+).

    Unlike ``capture_window_image`` (Quartz), this works for windows
    that are:

    * in another Space (e.g. fullscreen Safari while the user has
      switched to their terminal Space),
    * minimised to the Dock,
    * hidden behind opaque windows.

    Returns a BGR ``np.ndarray`` (compatible with ``cv2``) or ``None``
    on any failure (non-macOS, missing pyobjc-framework-ScreenCapture
    Kit, missing Screen Recording permission, window gone, timeout).
    Caller should fall back to ``capture_window_image`` then mss.

    ``target_size``: optional ``(width, height)`` in *logical* pixels.
    SCKit returns the image at native (physical) resolution, which is
    2x logical on Retina displays. Pass the window's logical bbox
    here to get an array the same size mss would have returned.
    """
    if not _is_macos():
        return None
    try:
        import ScreenCaptureKit  # type: ignore
        import Quartz  # type: ignore - reused for CGImage byte unpack
        import cv2  # type: ignore
    except ImportError:
        return None

    # Build (and cache) the filter+config for this windowID. Cached
    # objects are reused on subsequent ticks; we only rebuild on
    # window-id change or invalidate_sck_cache().
    if _SCK_CACHE["window_id"] != window_id or _SCK_CACHE["filter"] is None:
        content = _sck_get_shareable_content(timeout=timeout)
        if content is None:
            return None
        filter_, config = _sck_build_filter_and_config(window_id, content)
        if filter_ is None or config is None:
            return None
        _SCK_CACHE["window_id"] = window_id
        _SCK_CACHE["filter"] = filter_
        _SCK_CACHE["config"] = config

    filter_ = _SCK_CACHE["filter"]
    config = _SCK_CACHE["config"]

    event = threading.Event()
    box: dict = {"image": None, "error": None}

    def handler(image, error):  # noqa: ANN001 - pyobjc bridges types
        box["image"] = image
        box["error"] = error
        event.set()

    try:
        ScreenCaptureKit.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(  # type: ignore[attr-defined]
            filter_, config, handler,
        )
    except Exception as exc:  # noqa: BLE001
        # Older macOS without SCScreenshotManager (12.3-13) or
        # pyobjc that doesn't expose the symbol. Fall back.
        print(f"[sckit] SCScreenshotManager unavailable: {exc}")
        invalidate_sck_cache()
        return None

    if not event.wait(timeout):
        # Capture took too long; drop cached filter so the next call
        # re-enumerates (the window may have been closed).
        invalidate_sck_cache()
        return None

    if box["error"] is not None:
        # Most common: window was closed between enumeration and
        # capture. Drop cache and let the caller fall back.
        invalidate_sck_cache()
        return None

    cg_image = box["image"]
    if cg_image is None:
        invalidate_sck_cache()
        return None

    # Same CGImage -> ndarray conversion as capture_window_image.
    width = int(Quartz.CGImageGetWidth(cg_image))  # type: ignore[attr-defined]
    height = int(Quartz.CGImageGetHeight(cg_image))  # type: ignore[attr-defined]
    if width <= 0 or height <= 0:
        return None
    bytes_per_row = int(
        Quartz.CGImageGetBytesPerRow(cg_image)  # type: ignore[attr-defined]
    )
    data_provider = Quartz.CGImageGetDataProvider(cg_image)  # type: ignore[attr-defined]
    raw = Quartz.CGDataProviderCopyData(data_provider)  # type: ignore[attr-defined]
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size < bytes_per_row * height:
        return None
    arr = buf[: bytes_per_row * height].reshape((height, bytes_per_row // 4, 4))
    bgr = arr[:, :width, :3].copy()
    if target_size is not None and (width, height) != target_size:
        bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
    return bgr


def pick_mode_dialog() -> str:
    """Pop a tiny 2-button window asking how to select the capture
    region. Returns ``"window"``, ``"roi"``, or ``"cancel"``."""
    try:
        import tkinter as tk
    except ImportError:
        print(
            "[window-picker] tkinter not available - falling back to "
            "drag-ROI. Install with `brew install python-tk@3.11` (or "
            "the matching version for your Python) to get the picker."
        )
        return "roi"

    result = {"choice": "cancel"}

    def _close(c: str) -> None:
        result["choice"] = c
        root.destroy()

    root = tk.Tk()
    root.title("XocDia capture mode")
    try:
        root.eval("tk::PlaceWindow . center")  # macOS may ignore but harmless
    except tk.TclError:
        pass
    root.geometry("360x150")
    root.resizable(False, False)
    tk.Label(
        root,
        text="How would you like to choose the capture region?",
        font=("Helvetica", 13),
        wraplength=320,
    ).pack(pady=18)
    btns = tk.Frame(root)
    btns.pack()
    tk.Button(
        btns, text="Pick Window", width=14, height=2,
        command=lambda: _close("window"),
    ).pack(side="left", padx=8)
    tk.Button(
        btns, text="Drag ROI", width=14, height=2,
        command=lambda: _close("roi"),
    ).pack(side="left", padx=8)
    root.protocol("WM_DELETE_WINDOW", lambda: _close("cancel"))
    root.bind("<Escape>", lambda _e: _close("cancel"))
    root.mainloop()
    return result["choice"]


def pick_window_dialog(windows: List[WindowInfo]) -> Optional[WindowInfo]:
    """Show a Tk listbox to choose one of the supplied windows.
    Returns the chosen ``WindowInfo`` or ``None`` if the user
    cancelled/closed the dialog."""
    if not windows:
        return None
    try:
        import tkinter as tk
    except ImportError:
        print(
            "[window-picker] tkinter not available - cannot show window "
            "list. Install with `brew install python-tk@3.11`."
        )
        return None

    chosen: dict[str, Optional[WindowInfo]] = {"win": None}

    def _confirm() -> None:
        sel = lb.curselection()
        if sel:
            chosen["win"] = windows[int(sel[0])]
        root.destroy()

    root = tk.Tk()
    root.title("Select window to capture")
    root.geometry("560x440")
    tk.Label(
        root,
        text=f"Pick a window  ({len(windows)} visible)",
        font=("Helvetica", 12),
    ).pack(pady=(10, 4))
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10)
    sb = tk.Scrollbar(frame, orient="vertical")
    lb = tk.Listbox(
        frame,
        font=("Menlo", 11),
        height=16,
        yscrollcommand=sb.set,
        activestyle="dotbox",
    )
    sb.config(command=lb.yview)
    sb.pack(side="right", fill="y")
    lb.pack(side="left", fill="both", expand=True)
    for w in windows:
        lb.insert("end", w.label)
    lb.selection_set(0)
    lb.see(0)
    lb.bind("<Double-Button-1>", lambda _e: _confirm())
    lb.bind("<Return>", lambda _e: _confirm())
    tk.Button(root, text="Capture this window", command=_confirm).pack(pady=8)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()
    return chosen["win"]
