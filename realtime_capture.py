"""Real-time screen capture + detect + pipeline-analyze for Xoc Dia.

IMPROVEMENTS in this version:
- More robust ROI selection with visual feedback
- Better auto-clamp with adaptive retries
- Stabilized detection with frame buffering
- Enhanced error recovery
- Better window tracking
- Improved diagnostics

15-class schema. Round tracking is done here via a small state machine
driven by the ``timer`` and ``dice_*`` classes, because those are the only
stable temporal signals in the game UI.

Round lifecycle (one pass of the loop logs exactly once per round):

    IDLE
      |  (timer observed with value >= TIMER_NEW_ROUND_THRESHOLD)
      v
    NEW_ROUND  -> snapshot percent from scoreboard (scoreboard is stable here)
      |
      v
    BETTING    -> keep refreshing bets/counts while timer counts down
      |
      |  (dice_* detected)
      v
    RESULT     -> emit one log line with bets + counts + dice + percent,
      |          save frame + JSON to rounds/, then return to IDLE
      v
    IDLE

Hotkeys:
    r  select capture region by drag-select on full screen
       (auto-tightens to the detected UI bbox after selection)
    c  re-tighten the current region to the detected UI bbox
    s  save current preview frame to preview_capture.png
    d  toggle diagnostic overlay
    q  quit
"""

import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque

import cv2
import mss
import numpy as np

from detector import XocDiaDetector  # noqa: F401 - kept for external callers
from pipeline import BET_TYPES, GameAnalysisPipeline, GameState, PERCENT_ROW_ORDER
import window_picker


# Timer threshold that marks the start of a new round. The game resets the
# countdown to ~48s; we fire "new round" the first time we see any value
# at or above this threshold after a finished round.
TIMER_NEW_ROUND_THRESHOLD = 46


def resolve_weights(weights: str, fallback_pattern: str = "runs/**/weights/best.pt") -> str:
    path = Path(weights)
    if path.exists():
        return str(path)
    candidates = sorted(
        Path(".").glob(fallback_pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else weights


@dataclass
class Round:
    """One in-progress or just-finished round."""
    round_id: str                                   # timestamp-based id
    started_at: str                                 # ISO timestamp
    percent: Dict[str, Optional[str]] = field(default_factory=dict)
    # Per-area history of every sanitised percent reading observed
    # during the [46,48] timer window. Resolved into ``percent`` via
    # majority vote at finalisation. Multi-frame consensus protects us
    # against a single bad OCR frame in a 2-3 frame window.
    percent_history: Dict[str, List[str]] = field(default_factory=dict)
    bets: Dict[str, Dict[str, Optional[str]]] = field(
        default_factory=lambda: {k: {"total_bet": None, "total_count": None} for k in BET_TYPES}
    )
    dice_result: Optional[str] = None
    finalised_at: Optional[str] = None

    def update_bets(self, state: GameState) -> None:
        for bet_type, bet in state.bets.items():
            slot = self.bets.setdefault(
                bet_type, {"total_bet": None, "total_count": None},
            )
            if bet.total_bet:
                slot["total_bet"] = bet.total_bet
            if bet.total_count:
                slot["total_count"] = bet.total_count

    def update_percent(self, state: GameState) -> None:
        """Append every non-null sanitised percent reading to history.
        Resolution to a single value per area is deferred to
        ``finalise_percent`` so all in-window reads vote together."""
        for bet_type, bet in state.bets.items():
            if bet.percent:
                self.percent_history.setdefault(bet_type, []).append(bet.percent)

    def finalise_percent(self) -> None:
        """Collapse ``percent_history`` to a single value per area by
        majority vote, letting later good reads outvote a single-frame
        OCR error. On ties (e.g. exactly one good and one bad read)
        Counter.most_common's tie-break is implementation-defined, so we
        explicitly pick the first-seen reading via ``readings.index`` to
        match the previous "first-read wins" behaviour deterministically."""
        for bet_type, readings in self.percent_history.items():
            if not readings:
                continue
            counts = Counter(readings)
            top_count = counts.most_common(1)[0][1]
            tied = [r for r, c in counts.items() if c == top_count]
            # First-seen-among-tied wins.
            winner = min(tied, key=readings.index)
            self.percent[bet_type] = winner

    def to_dict(self) -> Dict[str, object]:
        return {
            "round_id": self.round_id,
            "started_at": self.started_at,
            "finalised_at": self.finalised_at,
            "dice_result": self.dice_result,
            "percent": dict(self.percent),
            "bets": dict(self.bets),
        }


class RoundTracker:
    """State machine that turns a stream of GameStates into round events."""

    def __init__(
        self,
        rounds_dir: Path,
        percent_row_order=None,
    ):
        self.rounds_dir = rounds_dir
        self.rounds_dir.mkdir(exist_ok=True)
        self.percent_row_order = list(percent_row_order or PERCENT_ROW_ORDER)

        # State is derived from (self.current, self.last_timer). No need
        # for a separate phase field; adding one tends to drift out of
        # sync with the real transitions.
        self.current: Optional[Round] = None
        self.last_timer: Optional[int] = None

    # -- core transitions --

    def ingest(self, state: GameState, frame: np.ndarray) -> Optional[Round]:
        """Feed one frame. Returns a finalised Round the moment it closes,
        otherwise ``None``. The caller owns logging and persistence."""
        timer = state.timer_int

        # Detect start of a new round: saw a fresh high timer value
        # (>= threshold). Reset any stale round we never closed.
        just_started = (
            timer is not None
            and timer >= TIMER_NEW_ROUND_THRESHOLD
            and (self.last_timer is None or self.last_timer < TIMER_NEW_ROUND_THRESHOLD)
        )
        if just_started:
            if self.current is not None and self.current.dice_result is None:
                print(
                    f"[WARN] Abandoning in-progress round {self.current.round_id} "
                    f"(no dice result observed before next round started)."
                )
            self._start_new_round(state)

        # We're inside an active round -> always refresh bets/counts.
        # Percent is only captured in the round-start window (timer >= 46)
        # when the scoreboard has just ticked over for the new round and
        # is visually stable. ``update_percent`` appends each non-null
        # reading to ``percent_history`` for the majority-vote in
        # ``finalise_percent``; the ``timer >= TIMER_NEW_ROUND_THRESHOLD``
        # guard is what stops it from accumulating reads outside the
        # window (``update_percent`` itself is no longer idempotent).
        if self.current is not None:
            self.current.update_bets(state)
            if timer is not None and timer >= TIMER_NEW_ROUND_THRESHOLD:
                self.current.update_percent(state)

        # Finalise when a dice result appears.
        if state.dice_result is not None and self.current is not None \
                and self.current.dice_result is None:
            finished = self._finalise_round(state, frame)
            self.last_timer = timer
            return finished

        self.last_timer = timer
        return None

    # -- helpers --

    def _start_new_round(self, state: GameState) -> None:
        now = datetime.now()
        self.current = Round(
            round_id=now.strftime("%Y%m%d_%H%M%S"),
            started_at=now.isoformat(timespec="seconds"),
        )
        # NOTE: do NOT call update_percent here. The caller (``ingest``)
        # already does it for the same frame inside the
        # ``timer >= TIMER_NEW_ROUND_THRESHOLD`` block. Calling it here
        # too would double-count the first frame's reading in
        # ``percent_history``, giving it 2x weight in the majority vote
        # at finalisation.

    def _finalise_round(self, state: GameState, frame: np.ndarray) -> Round:
        assert self.current is not None
        self.current.dice_result = state.dice_result
        self.current.finalised_at = datetime.now().isoformat(timespec="seconds")
        # Make sure any late bet updates are captured.
        self.current.update_bets(state)
        # Collapse the multi-frame percent history into a single value
        # per area before logging / saving.
        self.current.finalise_percent()
        self._save_round(self.current, frame)
        finished = self.current
        self.current = None
        return finished

    def _save_round(self, rd: Round, frame: np.ndarray) -> None:
        # JSON-only: skip PNG to keep the realtime loop light. If you ever
        # want the frame for debugging, call tools/visualize on a captured
        # dataset image instead.
        del frame  # intentionally unused
        base = self.rounds_dir / f"{rd.round_id}.json"
        # Collision-safe: if two rounds share the second, append -1/-2/...
        if base.exists():
            i = 1
            while (self.rounds_dir / f"{rd.round_id}-{i}.json").exists():
                i += 1
            base = self.rounds_dir / f"{rd.round_id}-{i}.json"
        with open(base, "w", encoding="utf-8") as f:
            json.dump(rd.to_dict(), f, indent=2, ensure_ascii=False)

    # -- presentation --

    def format_log(self, rd: Round) -> str:
        lines = ["=" * 60]
        lines.append(
            f"ROUND {rd.round_id} | Dice: {rd.dice_result or '-'}"
        )
        for bet_type in self.percent_row_order:
            slot = rd.bets.get(bet_type, {})
            tb = slot.get("total_bet") or "-"
            tc = slot.get("total_count") or "-"
            lines.append(f"  {bet_type:<8} total_bet={tb:<10} count={tc}")

        pct_parts = []
        for bet_type in self.percent_row_order:
            val = rd.percent.get(bet_type) or "-"
            pct_parts.append(f"{bet_type} {val}")
        lines.append("PERCENT: " + " | ".join(pct_parts))
        lines.append("=" * 60)
        return "\n".join(lines)


class RealtimeCapture:
    def __init__(
        self,
        weights: str = "runs/detect/xocdia/weights/best.pt",
        conf: float = 0.25,
        imgsz: int = 800,
        device=None,
        log_ocr_rejects: bool = False,
        debug_cells_dir: Optional[str] = None,
        preview_fps: float = 10.0,
        diag: bool = False,
        stabilize_frames: int = 3,  # NEW: number of frames to buffer for stability
        rounds_dir: Optional[Path] = None,
    ):
        resolved = resolve_weights(weights)
        print(f"Model weights: {resolved}")
        if debug_cells_dir:
            print(f"Debug cell crops -> {debug_cells_dir}/")

        self.pipeline = GameAnalysisPipeline(
            weights=resolved, conf=conf, imgsz=imgsz, device=device,
            log_ocr_rejects=log_ocr_rejects,
            debug_cells_dir=debug_cells_dir,
        )
        self.detector = self.pipeline.detector  # alias for overlay
        self.sct = mss.mss()

        self.monitor = {"top": 0, "left": 0, "width": 1280, "height": 800}
        self.preview_window = "XocDia Preview"
        # ``rounds_dir`` is overridable so the .app bundle can write to
        # ``~/Documents/XocDia/rounds`` instead of the bundle's CWD
        # (which is read-only). Defaults to ./rounds for repo dev.
        rd_path = rounds_dir if rounds_dir is not None else Path("rounds")
        print(f"Rounds output: {rd_path.resolve()}")
        self.tracker = RoundTracker(rd_path)
        self.last_state: Optional[GameState] = None
        self.last_frame: Optional[np.ndarray] = None
        self.running = True

        # NEW: Frame stability buffer - helps with intermittent detection issues
        self.stabilize_frames = max(1, stabilize_frames)
        self.detection_history: Deque[int] = deque(maxlen=self.stabilize_frames)
        self.stable_detection_threshold = max(1, self.stabilize_frames // 2)

        # Cap preview redraw rate. Detection still runs on its own
        # ``--interval`` schedule (default 1s); the preview only needs
        # to look smooth, not to drive detection. 10 FPS by default
        # cuts CPU drastically vs. the previous ~30 FPS busy loop.
        self.preview_fps = max(1.0, preview_fps)
        self._last_preview_render = 0.0
        self.diag = diag
        self.diag_overlay = False  # NEW: toggle for diagnostic overlay

        # Window-picker bookkeeping (PR #23). When the user picks a
        # specific application window, ``window_id`` is the macOS
        # CGWindowID we re-poll every ``window_refresh_interval`` s to
        # follow move/resize. Stays ``None`` for drag-ROI captures.
        self.window_id: Optional[int] = None
        self.window_refresh_interval: float = 5.0
        self._last_window_check: float = 0.0
        # Last known *outer* window bounds in screen coords. Used by
        # ``_refresh_window_bounds`` so we can distinguish "user moved
        # the window" (just translate ``self.monitor``) from "user
        # resized the window" (reset and re-clamp).
        self._window_bounds: Optional[Tuple[int, int, int, int]] = None
        
        # NEW: Capture health tracking
        self.capture_failures = 0
        self.max_capture_failures = 5
        self.last_successful_capture = time.time()

    # ---------- capture ----------
    def capture(self) -> Optional[np.ndarray]:
        """Capture a single frame.

        Capture path order in window-picker mode (PR #25):

        1. ``ScreenCaptureKit`` - macOS 14+, captures live frames even
           when the target window is on another Space (e.g. fullscreen
           Safari while the user is on the terminal Space). Requires
           ``pyobjc-framework-ScreenCaptureKit`` and a one-time Screen
           Recording permission grant.
        2. ``CGWindowListCreateImage`` (Quartz) - works on older macOS
           and when SCKit isn't installed. Limitation: returns a
           *stale snapshot* for windows in another Space.
        3. ``mss`` screen-region capture - last resort. Reads whatever
           is currently rendered at ``self.monitor``'s screen
           coordinates, so other windows on top will replace the game
           in the captured frame.

        Returns ``None`` after ``max_capture_failures`` consecutive
        exceptions so the loop can decide what to do.
        """
        try:
            if self.window_id is not None:
                target = (self.monitor["width"], self.monitor["height"])

                # 1. ScreenCaptureKit (cross-Space, fullscreen-safe)
                img = window_picker.screen_capture_kit_capture(
                    self.window_id, target_size=target,
                )
                if img is not None:
                    self.capture_failures = 0
                    self.last_successful_capture = time.time()
                    return img

                # 2. Quartz CGWindowListCreateImage (same-Space only)
                img = window_picker.capture_window_image(
                    self.window_id, target_size=target,
                )
                if img is not None:
                    self.capture_failures = 0
                    self.last_successful_capture = time.time()
                    return img

                # Both window-aware paths failed - fall through to mss
                print(
                    "[capture] SCKit + Quartz both unavailable, "
                    "falling back to mss screen-region capture"
                )

            img = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.capture_failures = 0
            self.last_successful_capture = time.time()
            return frame

        except Exception as exc:
            self.capture_failures += 1
            print(
                f"[capture] ERROR: {exc} (failures: "
                f"{self.capture_failures}/{self.max_capture_failures})"
            )

            if self.capture_failures >= self.max_capture_failures:
                print("[capture] Too many failures, please check ROI settings")
                return None

            time.sleep(0.1)
            return None

    def select_region_with_mouse(self) -> bool:
        """Improved ROI selection with visual feedback and validation."""
        try:
            full_monitor = self.sct.monitors[1]
            full_img = np.array(self.sct.grab(full_monitor))
            full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)

            # NEW: Add visual guide overlay
            overlay = full_img.copy()
            h, w = overlay.shape[:2]
            
            # Draw center crosshair
            cv2.line(overlay, (w//2, 0), (w//2, h), (0, 255, 0), 1)
            cv2.line(overlay, (0, h//2), (w, h//2), (0, 255, 0), 1)
            
            # Draw instruction text
            cv2.putText(
                overlay,
                "Drag to select the game window. Press ENTER when done, ESC to cancel.",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.putText(
                overlay,
                "TIP: Select a bit wider than the game for better detection",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            
            # Blend overlay
            alpha = 0.95
            display = cv2.addWeighted(overlay, alpha, full_img, 1-alpha, 0)

            print("=" * 60)
            print("REGION SELECTION MODE")
            print("=" * 60)
            print("1. Drag to select the game window area")
            print("2. Include a small margin around the game")
            print("3. Press ENTER/SPACE to confirm")
            print("4. Press ESC to cancel")
            print("=" * 60)
            
            x, y, w, h = cv2.selectROI(
                "Select Game Region", display, showCrosshair=True, fromCenter=False,
            )
            cv2.destroyWindow("Select Game Region")

            # NEW: Validate selection
            if w > 0 and h > 0:
                # Check minimum size
                if w < 400 or h < 300:
                    print(f"[WARNING] Selected region is very small ({w}x{h}). Recommended minimum: 400x300")
                    print("[WARNING] Detection may be unreliable. Consider selecting a larger area.")
                
                # Check aspect ratio (game is usually ~4:3 or 16:9)
                aspect = w / h if h > 0 else 0
                if aspect < 0.8 or aspect > 2.5:
                    print(f"[WARNING] Unusual aspect ratio {aspect:.2f}. Game window is typically 1.3-1.8")
                
                self.monitor = {
                    "top": int(full_monitor["top"] + y),
                    "left": int(full_monitor["left"] + x),
                    "width": int(w),
                    "height": int(h),
                }
                print(f"✓ Region set: {w}x{h} @ ({self.monitor['left']}, {self.monitor['top']})")
                
                # NEW: Show preview of selected region
                preview_frame = self.capture()
                if preview_frame is not None:
                    preview_resized = cv2.resize(preview_frame, (800, 600))
                    cv2.putText(
                        preview_resized,
                        "Selected Region Preview - Press any key to continue",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                    cv2.imshow("Region Preview", preview_resized)
                    cv2.waitKey(2000)  # Show for 2 seconds or until keypress
                    cv2.destroyWindow("Region Preview")
                
                return True
            
            print("Region selection cancelled (empty selection)")
            return False
            
        except Exception as exc:
            print(f"[ERROR] Region selection failed: {exc}")
            return False

    def auto_clamp_roi(
        self,
        clamp_imgsz: int = 1280,
        margin: float = 0.08,  # IMPROVED: slightly larger margin
        min_dets: int = 2,
        attempts: int = 5,  # IMPROVED: more attempts
        attempt_gap_s: float = 0.4,  # IMPROVED: longer gap between attempts
        warmup_s: float = 0.8,  # IMPROVED: longer warmup
        clamp_conf: float = 0.15,  # IMPROVED: lower confidence for initial detection
        min_area_ratio: float = 0.4,  # IMPROVED: more lenient
    ) -> bool:
        """Improved auto-clamp with better detection stability."""
        print(f"[auto-clamp] Starting ROI optimization (warmup {warmup_s}s)...")
        
        if warmup_s > 0:
            time.sleep(warmup_s)

        best_results = None
        best_n = -1
        best_frame_shape = None
        best_bbox = None
        
        print(f"[auto-clamp] Attempting {attempts} detection passes...")
        
        for i in range(max(1, attempts)):
            if i > 0 and attempt_gap_s > 0:
                time.sleep(attempt_gap_s)
                
            frame = self.capture()
            if frame is None:
                print(f"[auto-clamp] attempt {i+1}/{attempts}: capture failed")
                continue
                
            try:
                results = self.detector.model(
                    frame,
                    conf=clamp_conf,
                    iou=self.detector.iou,
                    device=self.detector.device,
                    imgsz=clamp_imgsz,
                    verbose=False,
                )[0]
            except Exception as exc:  # noqa: BLE001
                print(f"[auto-clamp] attempt {i+1}/{attempts}: YOLO failed: {exc!r}")
                continue
                
            n = 0 if results.boxes is None else len(results.boxes)
            
            # NEW: Show detection quality
            quality = "excellent" if n >= 12 else "good" if n >= 6 else "fair" if n >= 3 else "poor"
            print(f"[auto-clamp] attempt {i+1}/{attempts}: {n} detection(s) [{quality}]")
            
            if n > best_n:
                best_n = n
                best_results = results
                best_frame_shape = frame.shape[:2]
                
                # NEW: Calculate and store bbox for this attempt
                if results.boxes is not None and len(results.boxes) > 0:
                    H, W = frame.shape[:2]
                    xyxy = results.boxes.xyxy.cpu().numpy()
                    x1 = float(xyxy[:, 0].min())
                    y1 = float(xyxy[:, 1].min())
                    x2 = float(xyxy[:, 2].max())
                    y2 = float(xyxy[:, 3].max())
                    best_bbox = (x1, y1, x2, y2, W, H)
            
            # Early-exit with higher threshold for quality
            if n >= 10:  # IMPROVED: wait for better detection
                print(f"[auto-clamp] Early exit - excellent detection quality")
                break

        if best_results is None or best_n < min_dets:
            print(
                f"[auto-clamp] ✗ Best attempt had {max(0, best_n)} detection(s) "
                f"(need >={min_dets}). Keeping current ROI."
            )
            print(f"[auto-clamp] TIPS:")
            print(f"  • Make sure the game window is visible and not minimized")
            print(f"  • Try selecting a larger region around the game")
            print(f"  • Check if the game is in betting or result phase")
            return False

        # NEW: Use stored bbox
        if best_bbox is None:
            print("[auto-clamp] ✗ No valid bounding box computed")
            return False
            
        x1, y1, x2, y2, W, H = best_bbox
        
        bw = x2 - x1
        bh = y2 - y1
        x1 -= bw * margin
        y1 -= bh * margin
        x2 += bw * margin
        y2 += bh * margin

        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(W, x2))
        y2 = int(min(H, y2))
        new_w = x2 - x1
        new_h = y2 - y1
        
        if new_w <= 0 or new_h <= 0:
            print("[auto-clamp] ✗ Invalid bbox dimensions")
            return False

        # Safety check
        old_area = float(self.monitor["width"]) * float(self.monitor["height"])
        new_area = float(new_w) * float(new_h)
        
        if old_area > 0 and (new_area / old_area) < min_area_ratio:
            ratio_pct = (new_area / old_area) * 100
            print(
                f"[auto-clamp] ⚠ Proposed bbox is {ratio_pct:.0f}% of source "
                f"(min {int(min_area_ratio * 100)}%). This may be a partial detection."
            )
            print(f"[auto-clamp] Keeping current ROI for safety.")
            print(f"[auto-clamp] TIP: Ensure the full game UI is visible during clamp")
            return False

        before = dict(self.monitor)
        self.monitor = {
            "top": int(self.monitor["top"] + y1),
            "left": int(self.monitor["left"] + x1),
            "width": int(new_w),
            "height": int(new_h),
        }
        
        print("=" * 60)
        print(f"[auto-clamp] ✓ ROI OPTIMIZED")
        print(f"  Before: {before['width']}x{before['height']} @ ({before['left']},{before['top']})")
        print(f"  After:  {new_w}x{new_h} @ ({self.monitor['left']},{self.monitor['top']})")
        print(f"  Using:  {best_n} detection(s)")
        print(f"  Reduction: {100 * (1 - new_area/old_area):.1f}%")
        print("=" * 60)
        return True

    # ---------- window-picker capture ----------
    def pick_window_via_dialog(self) -> bool:
        """Enumerate windows via Quartz, pop a Tk listbox to choose
        one, and set ``self.monitor`` / ``self.window_id`` to the
        selected window's bbox.

        Returns ``True`` when a window was picked, ``False`` on
        cancel / empty list.
        """
        windows = window_picker.list_windows()
        if not windows:
            print(
                "[window-picker] no windows available (Quartz missing or "
                "non-macOS). Falling back to drag-ROI."
            )
            return False
        chosen = window_picker.pick_window_dialog(windows)
        if chosen is None:
            print("[window-picker] cancelled.")
            return False
        x, y, w, h = chosen.bbox
        self.monitor = {
            "top": int(y), "left": int(x),
            "width": int(w), "height": int(h),
        }
        self.window_id = chosen.window_id
        self._window_bounds = (int(x), int(y), int(w), int(h))
        self._last_window_check = time.time()
        # Drop any cached SCKit filter built for a previous window
        # so the next capture rebuilds for the new windowID.
        window_picker.invalidate_sck_cache()
        print(
            f"[window-picker] capturing '{chosen.app_name} - "
            f"{chosen.title or '(untitled)'}' "
            f"({w}x{h} @ {x},{y}) id={chosen.window_id}"
        )
        return True

    def _refresh_window_bounds(self) -> None:
        """If we are tracking a specific window, re-fetch its bbox and
        update ``self.monitor`` to follow it."""
        if self.window_id is None:
            return
        bounds = window_picker.get_window_bounds(self.window_id)
        if bounds is None:
            print(
                f"[window-picker] window id={self.window_id} no longer "
                f"visible; keeping last-known bounds. Press 'r' to pick "
                f"another window."
            )
            self.window_id = None
            return
        x, y, w, h = bounds
        prev = self._window_bounds
        self._window_bounds = (int(x), int(y), int(w), int(h))
        if prev is not None and (x, y, w, h) == prev:
            return
        self.monitor = {
            "top": int(y), "left": int(x),
            "width": int(w), "height": int(h),
        }
        if prev is not None:
            kind = "resized" if (w, h) != prev[2:] else "moved"
            print(
                f"[window-picker] tracked window {kind} -> "
                f"{w}x{h} @ ({x},{y})"
            )
            # On resize, drop the SCKit cache so the next capture
            # rebuilds with the new output dimensions. Pure moves
            # don't change capture size so the cache stays valid.
            if (w, h) != prev[2:]:
                window_picker.invalidate_sck_cache()

    def select_capture_source(self, mode: str = "auto") -> bool:
        """Top-level entry-point used by ``start()`` and the ``r``
        hotkey. ``mode``:

        * ``"auto"``    - pop the 2-button dialog and dispatch.
        * ``"window"``  - skip the dialog, go straight to window list.
        * ``"roi"``     - skip the dialog, go straight to drag ROI.

        Returns ``True`` when a region was selected (caller can then
        run auto-clamp), ``False`` on cancel.
        """
        if mode == "auto":
            choice = window_picker.pick_mode_dialog()
        else:
            choice = mode
        if choice == "window":
            picked = self.pick_window_via_dialog()
            if picked:
                return True
            # Fall through to drag-ROI when window list was empty or
            # the user cancelled
            return self.select_region_with_mouse()
        if choice == "roi":
            # Clear any previous window tracking
            self.window_id = None
            self._window_bounds = None
            window_picker.invalidate_sck_cache()
            return self.select_region_with_mouse()
        # cancel / unexpected
        return False

    # ---------- main loop ----------
    def start(
        self,
        interval: float = 1.0,
        show_preview: bool = True,
        auto_roi: bool = True,
        auto_clamp: bool = True,
        capture_mode: str = "auto",
    ) -> None:
        roi_set = False
        effective_mode = capture_mode
        if not show_preview and capture_mode != "window":
            effective_mode = "window"
            
        if auto_roi:
            roi_set = self.select_capture_source(mode=effective_mode)
            
        # Persist the flag so the 'r' hotkey honours --no-auto-clamp
        self.auto_clamp = auto_clamp
        self._capture_mode = effective_mode
        
        # Skip auto-clamp in window-picker mode
        if auto_clamp and roi_set and self.window_id is None:
            self.auto_clamp_roi()
            
        print("=" * 60)
        print("REALTIME DETECTION STARTED")
        print("=" * 60)
        print("Hotkeys:")
        print("  r - Select new region")
        print("  c - Re-clamp current region")
        print("  d - Toggle diagnostic overlay")
        print("  s - Save current frame")
        print("  q - Quit")
        print("=" * 60)
        
        last_detect = 0.0
        preview_period = 1.0 / self.preview_fps
        idle_sleep = min(0.05, preview_period / 2.0)
        
        while self.running:
            try:
                now = time.time()
                detect_due = (now - last_detect) >= interval
                preview_due = show_preview and (
                    now - self._last_preview_render >= preview_period
                )
                
                if not (detect_due or preview_due):
                    time.sleep(idle_sleep)
                    continue

                frame = self.capture()
                if frame is None:
                    # NEW: Handle capture failures gracefully
                    if show_preview:
                        # Show error overlay
                        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            error_frame,
                            "CAPTURE ERROR - Press 'r' to reselect region",
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        )
                        cv2.imshow(self.preview_window, error_frame)
                        key = cv2.waitKey(1) & 0xFF
                        self._handle_hotkeys(key, None)
                    time.sleep(0.5)
                    continue
                    
                self.last_frame = frame

                if detect_due:
                    state = self.pipeline.analyze(frame)
                    self.last_state = state
                    last_detect = now
                    
                    # NEW: Track detection stability
                    n_dets = len(state.raw_detections)
                    self.detection_history.append(n_dets)
                    avg_dets = sum(self.detection_history) / len(self.detection_history)
                    
                    finished = self.tracker.ingest(state, frame)
                    
                    if self.diag:
                        n_dets = len(state.raw_detections)
                        if finished is not None:
                            cur_id = finished.round_id
                        elif self.tracker.current is not None:
                            cur_id = self.tracker.current.round_id
                        else:
                            cur_id = "-"
                        
                        # NEW: Enhanced diagnostics
                        stability = "STABLE" if avg_dets >= self.stable_detection_threshold else "UNSTABLE"
                        print(
                            f"[diag] phase={state.phase} timer={state.timer} "
                            f"dice={state.dice_result} dets={n_dets} avg={avg_dets:.1f} "
                            f"status={stability} round={cur_id} "
                            f"mon={self.monitor['width']}x{self.monitor['height']}"
                            f"@({self.monitor['left']},{self.monitor['top']})"
                        )
                        
                    if finished is not None:
                        print(self.tracker.format_log(finished))

                if preview_due:
                    self._render_preview(frame)
                    self._last_preview_render = now
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_hotkeys(key, frame)

                # Follow window move/resize
                if (
                    self.window_id is not None
                    and now - self._last_window_check
                    >= self.window_refresh_interval
                ):
                    self._refresh_window_bounds()
                    self._last_window_check = now
                    
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Loop error: {exc}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)

        if show_preview:
            cv2.destroyAllWindows()
        
        print("=" * 60)
        print("DETECTION STOPPED")
        print("=" * 60)

    # ---------- rendering ----------
    def _render_preview(self, frame: np.ndarray) -> None:
        """Improved preview with better visual feedback."""
        preview = frame.copy()
        h, w = preview.shape[:2]

        # NEW: Detection stability indicator
        if len(self.detection_history) > 0:
            avg_dets = sum(self.detection_history) / len(self.detection_history)
            is_stable = avg_dets >= self.stable_detection_threshold
            
            # Status bar background
            cv2.rectangle(preview, (0, 0), (w, 90), (40, 40, 40), -1)
            
            # Region info
            cv2.putText(
                preview,
                f"Region: {self.monitor['width']}x{self.monitor['height']} "
                f"@ ({self.monitor['left']}, {self.monitor['top']})",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )
            
            # Detection status
            status_color = (0, 255, 0) if is_stable else (0, 165, 255)
            status_text = f"Detection: {'STABLE' if is_stable else 'UNSTABLE'} (avg {avg_dets:.1f})"
            cv2.putText(
                preview, status_text,
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1,
            )
            
            # Hotkeys
            cv2.putText(
                preview,
                "Hotkeys: [r]egion | [c]lamp | [d]iag | [s]ave | [q]uit",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
            )

        # Draw detections
        if self.last_state:
            preview = self.detector.annotate(preview, self.last_state.raw_detections)
            state = self.last_state
            cur = self.tracker.current
            active_id = cur.round_id if cur else "-"
            
            # Game state info
            info_y = max(95, h - 60)
            cv2.rectangle(preview, (0, info_y - 5), (w, h), (40, 40, 40), -1)
            
            info = (
                f"Phase: {state.phase.upper()} | "
                f"Round: {active_id} | "
                f"Timer: {state.timer or '-'} | "
                f"Dice: {state.dice_result or '-'}"
            )
            cv2.putText(
                preview, info,
                (10, info_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            
            # NEW: Diagnostic overlay (toggle with 'd')
            if self.diag_overlay:
                self._render_diagnostic_overlay(preview, state)

        cv2.imshow(self.preview_window, preview)

    def _render_diagnostic_overlay(self, frame: np.ndarray, state: GameState) -> None:
        """Render detailed diagnostic information."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (w - 350, 100), (w - 10, h - 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 120
        line_height = 20
        
        cv2.putText(frame, "=== DIAGNOSTICS ===", (w - 340, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += line_height * 2
        
        # Detection counts by category
        groups = self.detector.group_by_category(state.raw_detections)
        for cat, dets in groups.items():
            if dets:
                cv2.putText(frame, f"{cat}: {len(dets)}", (w - 340, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y += line_height
        
        y += line_height
        
        # Bet states
        cv2.putText(frame, "Bets:", (w - 340, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += line_height
        
        for bet_type, bet in state.bets.items():
            status = "✓" if bet.total_bet and bet.total_count else "✗"
            cv2.putText(frame, f"{status} {bet_type[:6]}: {bet.total_bet or '-'}/{bet.total_count or '-'}",
                       (w - 340, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y += line_height - 2

    def _handle_hotkeys(self, key: int, frame: Optional[np.ndarray]) -> None:
        """Enhanced hotkey handling."""
        if key == ord("q"):
            self.running = False
            print("Quit requested")
        elif key == ord("r"):
            print("\n" + "=" * 60)
            print("REGION RESELECTION")
            print("=" * 60)
            mode = getattr(self, "_capture_mode", "auto")
            changed = self.select_capture_source(mode=mode)
            if (
                changed
                and getattr(self, "auto_clamp", True)
                and self.window_id is None
            ):
                self.auto_clamp_roi()
        elif key == ord("c"):
            if self.window_id is not None:
                print(
                    "[auto-clamp] Skipped: window-picker mode. "
                    "Press 'r' to switch to drag-ROI for manual clamp."
                )
            else:
                print("\n" + "=" * 60)
                print("MANUAL CLAMP REQUESTED")
                print("=" * 60)
                self.auto_clamp_roi()
        elif key == ord("d"):
            self.diag_overlay = not self.diag_overlay
            status = "ON" if self.diag_overlay else "OFF"
            print(f"[preview] Diagnostic overlay: {status}")
        elif key == ord("s"):
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"✓ Saved: {filename}")
            else:
                print("✗ No frame available to save")


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", default="runs/detect/xocdia/weights/best.pt")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Detection interval in seconds (default: 1.0)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Detection confidence threshold (default: 0.25). "
                            "Empirically tools/diag_detection.py shows 0.25 "
                            "recovers 5-7 extra cells per frame (mostly "
                            "percent_cell + total_*_cell) without false "
                            "positives at imgsz=800.")
    parser.add_argument("--imgsz", type=int, default=800,
                       help="Inference image size (default: 800)")
    parser.add_argument("--device", default=None,
                       help="Device to run inference on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--no-preview", action="store_true",
                       help="Disable preview window")
    parser.add_argument("--no-auto-roi", action="store_true",
                       help="Skip ROI selection at startup")
    parser.add_argument("--no-auto-clamp", action="store_true",
                       help="Skip automatic ROI optimization")
    parser.add_argument("--log-ocr-rejects", action="store_true",
                       help="Log OCR rejections for debugging")
    parser.add_argument("--debug-save-cells", nargs="?", const="debug_cells",
                       default=None, metavar="DIR",
                       help="Save cell crops for debugging")
    parser.add_argument("--preview-fps", type=float, default=10.0,
                       help="Preview window refresh rate (default: 10)")
    parser.add_argument("--diag", action="store_true",
                       help="Enable diagnostic output")
    parser.add_argument("--capture-mode", choices=("auto", "window", "roi"),
                       default="auto",
                       help="Capture mode: auto (dialog), window (picker), or roi (drag)")
    parser.add_argument("--stabilize-frames", type=int, default=3,
                       help="Number of frames to buffer for detection stability (default: 3)")
    parser.add_argument("--rounds-dir", default=None,
                       help="Output directory for round JSON files "
                            "(default: ./rounds in repo, ~/Documents/XocDia/rounds in .app)")
    return parser.parse_args()


def main() -> int:
    """Programmatic entry point. Returns process exit code so ``app_main``
    can wrap this with extra setup (logging, default paths) without
    duplicating the CLI plumbing."""
    args = _parse_args()
    rounds_dir = Path(args.rounds_dir) if args.rounds_dir else None
    cap = RealtimeCapture(
        weights=args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device,
        log_ocr_rejects=args.log_ocr_rejects,
        debug_cells_dir=args.debug_save_cells,
        preview_fps=args.preview_fps,
        diag=args.diag,
        stabilize_frames=args.stabilize_frames,
        rounds_dir=rounds_dir,
    )
    cap.start(
        interval=args.interval,
        show_preview=not args.no_preview,
        auto_roi=not args.no_auto_roi,
        auto_clamp=not args.no_auto_clamp,
        capture_mode=args.capture_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())