"""macOS window enumeration + Tkinter dialogs for capture-source selection.

Exposes three small helpers used by ``realtime_capture.py``:

* ``list_windows()``       - enumerate visible top-level windows (Quartz)
* ``get_window_bounds()``  - re-fetch bounds of a specific window by id
* ``pick_mode_dialog()``   - 2-button "Pick Window" / "Drag ROI" picker
* ``pick_window_dialog()`` - listbox to choose one window

Everything degrades gracefully on non-macOS or when ``pyobjc-framework-
Quartz`` isn't installed: ``list_windows()`` returns ``[]`` and the
caller falls back to the existing drag-ROI flow.

PR #23 motivation
-----------------
Drag-ROI lets users accidentally drag past the game window into their
own terminal, which then becomes part of the YOLO input frame and tanks
detection rates. Picking a specific window (Safari, Chrome, the
standalone game player, etc.) eliminates that whole class of mistake
and lets the script auto-follow the window when the user moves or
resizes it.
"""

from __future__ import annotations

import platform
from typing import List, NamedTuple, Optional, Tuple


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


def pick_mode_dialog() -> str:
    """Pop a tiny 2-button window asking how to select the capture
    region. Returns ``"window"``, ``"roi"``, or ``"cancel"``."""
    try:
        import tkinter as tk
    except ImportError:
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
