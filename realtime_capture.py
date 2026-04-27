"""Real-time screen capture + detect + pipeline-analyze for Xoc Dia.

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
    s  save current preview frame to preview_capture.png
    q  quit
"""

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mss
import numpy as np

from detector import XocDiaDetector  # noqa: F401 - kept for external callers
from pipeline import BET_TYPES, GameAnalysisPipeline, GameState, PERCENT_ROW_ORDER
from window_capture import WindowTracker


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
        conf: float = 0.4,
        imgsz: int = 800,
        device=None,
        log_ocr_rejects: bool = False,
        debug_cells_dir: Optional[str] = None,
        window_title: Optional[str] = None,
        window_owner: Optional[str] = None,
        preview_fps: float = 10.0,
        diag: bool = False,
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
        self.tracker = RoundTracker(Path("rounds"))
        self.last_state: Optional[GameState] = None
        self.last_frame: Optional[np.ndarray] = None
        self.running = True

        # Cap preview redraw rate. Detection still runs on its own
        # ``--interval`` schedule (default 1s); the preview only needs
        # to look smooth, not to drive detection. 10 FPS by default
        # cuts CPU drastically vs. the previous ~30 FPS busy loop.
        self.preview_fps = max(1.0, preview_fps)
        self._last_preview_render = 0.0
        self.diag = diag

        # Optional macOS window tracker. When set we re-resolve the
        # window bounds every couple of seconds so the capture box
        # follows the user's window across moves / resizes.
        self.window_tracker: Optional[WindowTracker] = None
        if window_title or window_owner:
            from window_capture import is_macos
            if not is_macos():
                print(
                    "[window-capture] --window-title/--window-owner are "
                    "macOS-only; ignoring and falling back to manual "
                    "ROI selection."
                )
            else:
                self.window_tracker = WindowTracker(
                    title_substring=window_title,
                    owner_substring=window_owner,
                )
                match = self.window_tracker.initial_resolve()
                if match is None:
                    print(
                        f"[window-capture] No window matched title="
                        f"{window_title!r} owner={window_owner!r}; "
                        f"falling back to manual ROI selection."
                    )
                    self.window_tracker = None
                else:
                    self.monitor = dict(match.monitor)
                    print(
                        f"[window-capture] Tracking window "
                        f"{match.owner!r} - {match.title!r} at {self.monitor}"
                    )

    # ---------- capture ----------
    def capture(self) -> np.ndarray:
        img = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def select_region_with_mouse(self) -> None:
        full_monitor = self.sct.monitors[1]
        full_img = np.array(self.sct.grab(full_monitor))
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)

        print("Drag to select game region, then press ENTER/SPACE. ESC to cancel.")
        x, y, w, h = cv2.selectROI(
            "Select Game Region", full_img, showCrosshair=True, fromCenter=False,
        )
        cv2.destroyWindow("Select Game Region")

        if w > 0 and h > 0:
            self.monitor = {
                "top": int(full_monitor["top"] + y),
                "left": int(full_monitor["left"] + x),
                "width": int(w),
                "height": int(h),
            }
            print(f"Updated region: {self.monitor}")
        else:
            print("Region selection cancelled.")

    # ---------- main loop ----------
    def start(self, interval: float = 1.0, show_preview: bool = True) -> None:
        print("Starting real-time detection. Hotkeys: r / s / q")
        last_detect = 0.0
        preview_period = 1.0 / self.preview_fps
        # Sleep granularity. Has to be small enough that hotkeys feel
        # responsive but large enough to keep idle CPU low.
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

                # Refresh the capture region from the window tracker
                # before grabbing the frame. Cheap most of the time
                # (Quartz lookup is throttled by ``poll_interval``).
                if self.window_tracker is not None:
                    mon = self.window_tracker.current_monitor()
                    if mon is not None:
                        self.monitor = dict(mon)

                frame = self.capture()
                self.last_frame = frame

                if detect_due:
                    state = self.pipeline.analyze(frame)
                    self.last_state = state
                    last_detect = now
                    finished = self.tracker.ingest(state, frame)
                    if self.diag:
                        n_dets = len(getattr(state, "raw_detections", []) or [])
                        cur_id = self.tracker.current.round_id if self.tracker.current else "-"
                        print(
                            f"[diag] phase={state.phase} timer={state.timer} "
                            f"dice={state.dice_result} dets={n_dets} "
                            f"round={cur_id} mon={self.monitor['width']}x"
                            f"{self.monitor['height']}@({self.monitor['left']},"
                            f"{self.monitor['top']})"
                        )
                    if finished is not None:
                        print(self.tracker.format_log(finished))

                if preview_due:
                    self._render_preview(frame)
                    self._last_preview_render = now
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_hotkeys(key, frame)
            except KeyboardInterrupt:
                print("Stopped.")
                break
            except Exception as exc:  # noqa: BLE001 - visibility in realtime loop
                print(f"Loop error: {exc}")
                time.sleep(0.2)

        if show_preview:
            cv2.destroyAllWindows()

    # ---------- rendering ----------
    def _render_preview(self, frame: np.ndarray) -> None:
        preview = frame.copy()
        h, _w = preview.shape[:2]

        cv2.putText(
            preview,
            f"Region left={self.monitor['left']} top={self.monitor['top']} "
            f"w={self.monitor['width']} h={self.monitor['height']}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        cv2.putText(
            preview,
            "Hotkeys: r=region | s=save frame | q=quit",
            (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        if self.last_state:
            preview = self.detector.annotate(preview, self.last_state.raw_detections)
            state = self.last_state
            cur = self.tracker.current
            active_id = cur.round_id if cur else "-"
            info = (
                f"phase={state.phase} round={active_id} "
                f"timer={state.timer} dice={state.dice_result}"
            )
            cv2.putText(
                preview, info,
                (10, max(70, h - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        cv2.imshow(self.preview_window, preview)

    def _handle_hotkeys(self, key: int, frame: np.ndarray) -> None:
        if key == ord("q"):
            self.running = False
        elif key == ord("r"):
            # The window tracker re-resolves the monitor every poll
            # interval (~2s) and would silently overwrite the user's
            # manual selection. Drop the tracker the moment the user
            # asks for a manual region - they've explicitly chosen to
            # override window tracking.
            if self.window_tracker is not None:
                print(
                    "[window-capture] Manual region selected; disabling "
                    "window tracker for this session."
                )
                self.window_tracker = None
            self.select_region_with_mouse()
        elif key == ord("s"):
            cv2.imwrite("preview_capture.png", frame)
            print("Saved preview_capture.png")


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/detect/xocdia/weights/best.pt")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--log-ocr-rejects",
        action="store_true",
        help="Print one [OCR-REJECT] line per cell whose OCR text "
             "didn't pass the per-class sanitiser. Useful when tuning "
             "label tightness or debugging weird log values.",
    )
    parser.add_argument(
        "--debug-save-cells",
        nargs="?",
        const="debug_cells",
        default=None,
        metavar="DIR",
        help="Dump every cell crop fed to the OCR (raw + preprocessed "
             "PNG) plus the OCR/sanitised text into DIR (default: "
             "'debug_cells'). Use this to collect real bbox crops for "
             "tuning preprocessing offline. Off by default - has I/O "
             "overhead, only enable for debug runs.",
    )
    parser.add_argument(
        "--window-title",
        default=None,
        metavar="SUBSTRING",
        help="macOS only: track an on-screen window whose title "
             "contains SUBSTRING (case-insensitive). The capture box "
             "follows the window across moves and resizes. Falls back "
             "to manual ROI selection if no window matches.",
    )
    parser.add_argument(
        "--window-owner",
        default=None,
        metavar="SUBSTRING",
        help="macOS only: combine with (or use instead of) "
             "--window-title to filter by the owning app's name "
             "(e.g. 'Safari', 'Chrome'). Useful when multiple apps "
             "have similar window titles.",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="macOS only: print the list of on-screen windows "
             "(owner + title) and exit. Use this to discover the "
             "right --window-title / --window-owner values.",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        default=10.0,
        help="Cap preview redraw rate (default: 10 FPS). Lower values "
             "reduce CPU on slow machines. Detection still runs on its "
             "own --interval schedule independently.",
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="Print one diagnostic line per detection tick "
             "(phase, timer, dice_result, det count, monitor bounds). "
             "Use this to verify detection is firing and YOLO is "
             "returning boxes when round summaries stop appearing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.list_windows:
        from window_capture import is_macos, list_windows
        if not is_macos():
            print("--list-windows is only supported on macOS.")
            raise SystemExit(1)
        for w in list_windows():
            owner = w.get("kCGWindowOwnerName", "")
            title = w.get("kCGWindowName", "") or ""
            bounds = w.get("kCGWindowBounds") or {}
            try:
                rect = (
                    f"{int(bounds['Width'])}x{int(bounds['Height'])} "
                    f"@ ({int(bounds['X'])},{int(bounds['Y'])})"
                )
            except (KeyError, TypeError, ValueError):
                rect = "?"
            print(f"  [{owner}] {title!r}  {rect}")
        raise SystemExit(0)

    cap = RealtimeCapture(
        weights=args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device,
        log_ocr_rejects=args.log_ocr_rejects,
        debug_cells_dir=args.debug_save_cells,
        window_title=args.window_title,
        window_owner=args.window_owner,
        preview_fps=args.preview_fps,
        diag=args.diag,
    )
    cap.start(interval=args.interval, show_preview=not args.no_preview)
