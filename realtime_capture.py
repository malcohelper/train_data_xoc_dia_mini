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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import mss
import numpy as np

from detector import XocDiaDetector  # noqa: F401 - kept for external callers
from pipeline import BET_TYPES, GameAnalysisPipeline, GameState, PERCENT_ROW_ORDER


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
        # Only overwrite slots we haven't captured yet; the scoreboard is
        # stable for the duration of the round so the first read is best.
        for bet_type, bet in state.bets.items():
            if bet.percent and not self.percent.get(bet_type):
                self.percent[bet_type] = bet.percent

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

        self.phase: str = "idle"      # idle | active
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

        # We're inside an active round -> keep updating bets/percent.
        if self.current is not None:
            self.current.update_bets(state)
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
        # Capture percent right away (scoreboard stable at round start).
        self.current.update_percent(state)

    def _finalise_round(self, state: GameState, frame: np.ndarray) -> Round:
        assert self.current is not None
        self.current.dice_result = state.dice_result
        self.current.finalised_at = datetime.now().isoformat(timespec="seconds")
        # Make sure any late bet updates are captured.
        self.current.update_bets(state)
        self._save_round(self.current, frame)
        finished = self.current
        self.current = None
        return finished

    def _save_round(self, rd: Round, frame: np.ndarray) -> None:
        base = self.rounds_dir / rd.round_id
        # Collision-safe: if two rounds somehow share the second, append
        # -1 / -2 / ... so we never overwrite an existing snapshot.
        if Path(f"{base}.png").exists() or Path(f"{base}.json").exists():
            i = 1
            while Path(f"{base}-{i}.png").exists() or Path(f"{base}-{i}.json").exists():
                i += 1
            base = self.rounds_dir / f"{rd.round_id}-{i}"
        cv2.imwrite(f"{base}.png", frame)
        with open(f"{base}.json", "w", encoding="utf-8") as f:
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
    ):
        resolved = resolve_weights(weights)
        print(f"Model weights: {resolved}")

        self.pipeline = GameAnalysisPipeline(
            weights=resolved, conf=conf, imgsz=imgsz, device=device,
        )
        self.detector = self.pipeline.detector  # alias for overlay
        self.sct = mss.mss()

        self.monitor = {"top": 0, "left": 0, "width": 1280, "height": 800}
        self.preview_window = "XocDia Preview"
        self.tracker = RoundTracker(Path("rounds"))
        self.last_state: Optional[GameState] = None
        self.last_frame: Optional[np.ndarray] = None
        self.running = True

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
        while self.running:
            try:
                frame = self.capture()
                self.last_frame = frame
                now = time.time()

                if now - last_detect >= interval:
                    state = self.pipeline.analyze(frame)
                    self.last_state = state
                    last_detect = now
                    finished = self.tracker.ingest(state, frame)
                    if finished is not None:
                        print(self.tracker.format_log(finished))

                if show_preview:
                    self._render_preview(frame)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_hotkeys(key, frame)

                time.sleep(0.03)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cap = RealtimeCapture(
        weights=args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device,
    )
    cap.start(interval=args.interval, show_preview=not args.no_preview)
