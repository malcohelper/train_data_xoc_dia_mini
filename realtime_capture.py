"""Real-time screen capture + detect + pipeline-analyze for Xoc Dia.

Rewritten for the 17-class single-stage schema:
- Uses ``XocDiaDetector`` + ``GameAnalysisPipeline`` (no sub_model).
- Overlay rendering uses shared ``classes.COLORS`` + detector.annotate.
- When a new round id is detected, the frame and GameState are persisted
  into ``rounds/<round_id>_<timestamp>.(png|json)`` like before.

Hotkeys:
    r  select capture region by drag-select on full screen
    s  save current preview frame to preview_capture.png
    q  quit
"""

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np

from detector import XocDiaDetector  # noqa: F401 - kept for external callers
from pipeline import GameAnalysisPipeline


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
        self.last_round_id = None
        self.last_state = None
        self.last_frame = None
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
    def start(self, interval: float = 2.0, show_preview: bool = True) -> None:
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
                    self._handle_state(state, frame)

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

    # ---------- state handling ----------
    def _handle_state(self, state, frame: np.ndarray) -> None:
        if state.round_id and state.round_id != self.last_round_id:
            print("=" * 60)
            print(f"NEW ROUND {state.round_id} | phase={state.phase} timer={state.timer}")
            print(f"Dice  : {state.dice_result}")
            for bet_type, bet in state.bets.items():
                pct = bet.percent or "-"
                tb = bet.total_bet or "-"
                tc = bet.total_count or "-"
                print(
                    f"  {bet_type:<8} percent={pct:<5} "
                    f"total_bet={tb:<8} count={tc}"
                )
            print("=" * 60)
            self.last_round_id = state.round_id
            self._save_round(state, frame)

    def _save_round(self, state, frame: np.ndarray) -> None:
        rounds_dir = Path("rounds")
        rounds_dir.mkdir(exist_ok=True)
        round_id = (state.round_id or "unknown").replace("#", "")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = rounds_dir / f"{round_id}_{ts}"
        cv2.imwrite(f"{base}.png", frame)
        payload = state.to_dict() if hasattr(state, "to_dict") else (
            asdict(state) if is_dataclass(state) else {}
        )
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---------- rendering ----------
    def _render_preview(self, frame: np.ndarray) -> None:
        preview = frame.copy()
        h, w = preview.shape[:2]

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
            info = (
                f"phase={state.phase} round={state.round_id} timer={state.timer} "
                f"dice={state.dice_result}"
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
    parser.add_argument("--interval", type=float, default=2.0)
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
