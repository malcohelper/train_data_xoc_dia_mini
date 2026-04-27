"""macOS window-targeted capture helpers.

``realtime_capture.py`` historically asked the user to drag-select a
region on the full screen. That ROI is fixed in screen coordinates, so
moving the game window invalidates it and the user has to reselect.

On macOS we can do better: query Quartz' window server for the bounds
of a specific app window by title substring, and refresh those bounds
periodically. This module exposes a single lookup helper plus a tiny
"sticky" tracker that re-resolves the window every ``poll_interval``
seconds so the capture rectangle follows the window if the user
drags / resizes it.

Non-macOS callers should not import Quartz; they can either skip the
feature entirely or fall back to the manual ROI flow. We do this by
guarding the import inside the lookup function so the module itself
remains importable on Linux/Windows.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


def is_macos() -> bool:
    return sys.platform == "darwin"


@dataclass
class WindowMatch:
    """A live macOS window matched by title/owner."""
    owner: str
    title: str
    monitor: Dict[str, int]  # mss-compatible: {top, left, width, height}


def list_windows() -> List[Dict[str, object]]:
    """Return the list of on-screen windows from Quartz, or [] on
    non-macOS / when Quartz is unavailable."""
    if not is_macos():
        return []
    try:
        # Lazy import keeps Linux/Windows boxes happy even though
        # pyobjc-framework-Quartz isn't installed.
        from Quartz import (  # type: ignore[import-not-found]
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListExcludeDesktopElements,
            kCGWindowListOptionOnScreenOnly,
        )
    except ImportError:
        print(
            "[window-capture] pyobjc-framework-Quartz not installed; "
            "install via `pip install pyobjc-framework-Quartz` to enable "
            "--window-title."
        )
        return []
    opts = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    raw = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) or []
    return list(raw)


def find_window(
    title_substring: Optional[str] = None,
    owner_substring: Optional[str] = None,
) -> Optional[WindowMatch]:
    """Return the first on-screen window whose title and/or owner
    contains the given substrings (case-insensitive). When both are
    given, both must match. Returns ``None`` if no window matches.
    """
    title_q = (title_substring or "").lower()
    owner_q = (owner_substring or "").lower()
    if not title_q and not owner_q:
        return None

    for w in list_windows():
        title = str(w.get("kCGWindowName", "") or "").strip()
        owner = str(w.get("kCGWindowOwnerName", "") or "").strip()
        if title_q and title_q not in title.lower():
            continue
        if owner_q and owner_q not in owner.lower():
            continue
        bounds = w.get("kCGWindowBounds") or {}
        try:
            left = int(bounds["X"])
            top = int(bounds["Y"])
            width = int(bounds["Width"])
            height = int(bounds["Height"])
        except (KeyError, TypeError, ValueError):
            continue
        # Some windows (menu bar overlays, helper popups) report tiny
        # bounds - skip those.
        if width < 100 or height < 100:
            continue
        return WindowMatch(
            owner=owner, title=title,
            monitor={
                "top": top, "left": left,
                "width": width, "height": height,
            },
        )
    return None


class WindowTracker:
    """Re-resolves the target window every ``poll_interval`` seconds.

    The realtime loop calls ``current_monitor()`` on every capture; if
    the cached monitor is fresh, returns it directly. Otherwise re-runs
    the Quartz lookup. This keeps the capture rectangle aligned with
    the window even if the user moves or resizes it.

    If lookup fails after the window was previously found, the last
    known good monitor is returned (the user probably just minimised
    the window briefly).
    """

    def __init__(
        self,
        title_substring: Optional[str] = None,
        owner_substring: Optional[str] = None,
        poll_interval: float = 2.0,
    ):
        self.title_substring = title_substring
        self.owner_substring = owner_substring
        self.poll_interval = poll_interval
        self._last_match: Optional[WindowMatch] = None
        self._last_lookup: float = 0.0

    def initial_resolve(self) -> Optional[WindowMatch]:
        """Force one lookup at startup. Returns the match or ``None``."""
        match = find_window(self.title_substring, self.owner_substring)
        if match is not None:
            self._last_match = match
            self._last_lookup = time.time()
        return match

    def current_monitor(self) -> Optional[Dict[str, int]]:
        now = time.time()
        if now - self._last_lookup >= self.poll_interval:
            match = find_window(self.title_substring, self.owner_substring)
            self._last_lookup = now
            if match is not None:
                self._last_match = match
        return self._last_match.monitor if self._last_match else None
