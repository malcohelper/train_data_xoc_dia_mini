"""Per-frame game-analysis pipeline: YOLO detect -> OCR -> structured state.

Round boundaries and the full "log 1 line per round" behaviour live in
``realtime_capture.RoundTracker``; this module only reports what is visible
in a single frame. The tracker decides when to log.

Flow per frame:
    1. Run XocDiaDetector to get 15-class detections.
    2. Group by category (state / area / dice / cell).
    3. Read timer (state category) and dice_result (dice category).
    4. For each text cell:
       - total_bet_cell  / total_count_cell: geometrically assign to the
         area_* that contains the cell's center.
       - percent_cell: the 6 scoreboard-% cells live ABOVE the play area,
         not inside any area_* bbox, so map them by y-order using
         PERCENT_ROW_ORDER (configurable).
    5. OCR each assigned crop via ocr_engine.XocDiaOCR.
    6. Figure out the current phase:
         - any dice_* class visible   -> "result"
         - timer visible and > 0      -> "betting"
         - otherwise                  -> "transition"
    7. Return a GameState dataclass.

Example::

    from pipeline import GameAnalysisPipeline
    import cv2

    pipe = GameAnalysisPipeline(weights="runs/detect/xocdia/weights/best.pt")
    state = pipe.analyze(cv2.imread("frame.png"))
    print(state)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from classes import CLASS_NAME_TO_ID
from detector import Detection, XocDiaDetector

# Row order of the 6 percent_cell detections in the scoreboard, top to
# bottom. Keys are the short bet-type names used throughout the pipeline.
# If the in-game layout changes, only this list needs updating.
PERCENT_ROW_ORDER: List[str] = [
    "chan",
    "4_red",
    "4_white",
    "le",
    "3r_1w",
    "3w_1r",
]

# Short bet-type names (index-aligned with the six area_* classes).
BET_TYPES: List[str] = ["chan", "le", "4_red", "3w_1r", "3r_1w", "4_white"]

# Mapping from area_* class name -> short bet-type name.
AREA_TO_BET = {f"area_{k}": k for k in BET_TYPES}

# Mapping from dice_* class name -> short dice outcome code.
DICE_TO_OUTCOME = {
    "dice_4r": "4_red",
    "dice_4w": "4_white",
    "dice_3w1r": "3w_1r",
    "dice_3r1w": "3r_1w",
    "dice_2w2r": "2w_2r",
}


@dataclass
class BetState:
    bet_type: str
    total_bet: Optional[str] = None       # OCR'd string, e.g. "7.47M"
    total_count: Optional[str] = None     # OCR'd string, e.g. "299"
    percent: Optional[str] = None         # OCR'd string, e.g. "53%"
    area_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class GameState:
    phase: str = "unknown"                # betting / result / transition
    timer: Optional[str] = None
    dice_result: Optional[str] = None     # 4_red / 4_white / 3w_1r / 3r_1w / 2w_2r
    bets: Dict[str, BetState] = field(default_factory=dict)
    raw_detections: List[Detection] = field(default_factory=list)

    # --- derived helpers ---
    @property
    def timer_int(self) -> Optional[int]:
        if self.timer is None:
            return None
        try:
            return int(self.timer)
        except (ValueError, TypeError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "timer": self.timer,
            "dice_result": self.dice_result,
            "bets": {
                k: {
                    "total_bet": v.total_bet,
                    "total_count": v.total_count,
                    "percent": v.percent,
                }
                for k, v in self.bets.items()
            },
        }


class GameAnalysisPipeline:
    def __init__(
        self,
        weights: str = "runs/detect/xocdia/weights/best.pt",
        conf: float = 0.4,
        iou: float = 0.45,
        imgsz: int = 800,
        device: Optional[str] = None,
        ocr: Optional["object"] = None,  # injected OCR engine for tests
        percent_row_order: Optional[List[str]] = None,
    ):
        self.detector = XocDiaDetector(
            weights=weights, conf=conf, iou=iou, imgsz=imgsz, device=device,
        )
        self.percent_row_order = percent_row_order or PERCENT_ROW_ORDER
        self._ocr = ocr

    # -------- OCR lazy init --------
    @property
    def ocr(self):
        if self._ocr is None:
            # Local import so the pipeline module stays importable even
            # when PaddleOCR dependencies are missing (useful for tests).
            from ocr_engine import XocDiaOCR
            self._ocr = XocDiaOCR()
        return self._ocr

    # -------- public API --------
    def analyze(self, frame: np.ndarray) -> GameState:
        detections = self.detector.detect(frame)
        groups = self.detector.group_by_category(detections)

        state = GameState(raw_detections=detections)

        # 1) Pre-seed all 6 bet slots with their area bbox so downstream
        # code can rely on state.bets having the expected keys.
        areas = self._index_areas(groups["area"])
        for bet_type, area_det in areas.items():
            state.bets[bet_type] = BetState(
                bet_type=bet_type, area_bbox=area_det.bbox,
            )

        # 2) State-level fields (timer, dice)
        self._fill_state_fields(frame, state, groups)

        # 3) Text cells -> bet slots
        self._fill_bet_cells(frame, state, groups, areas)

        # 4) Derive phase.
        state.phase = self._infer_phase(state)

        return state

    def annotate(
        self,
        frame: np.ndarray,
        state: Optional[GameState] = None,
    ) -> np.ndarray:
        if state is None:
            state = self.analyze(frame)
        return self.detector.annotate(frame, state.raw_detections)

    # -------- internals --------
    @staticmethod
    def _index_areas(area_dets: List[Detection]) -> Dict[str, Detection]:
        """Return {bet_type -> best Detection for that area_* class}."""
        best: Dict[str, Detection] = {}
        for d in area_dets:
            bet_type = AREA_TO_BET.get(d.class_name)
            if bet_type is None:
                continue
            if bet_type not in best or d.conf > best[bet_type].conf:
                best[bet_type] = d
        return best

    def _fill_state_fields(
        self,
        frame: np.ndarray,
        state: GameState,
        groups: Dict[str, List[Detection]],
    ) -> None:
        state_dets = groups.get("state", [])
        timer_dets = [d for d in state_dets if d.class_name == "timer"]
        if timer_dets:
            det = max(timer_dets, key=lambda x: x.conf)
            text = self.ocr.read_number(self.detector.crop(frame, det))
            state.timer = text

        # Dice result (0 or 1 instance per frame, in the "dice" category).
        dice_dets = groups.get("dice", [])
        if dice_dets:
            det = max(dice_dets, key=lambda x: x.conf)
            state.dice_result = DICE_TO_OUTCOME.get(det.class_name)

    def _fill_bet_cells(
        self,
        frame: np.ndarray,
        state: GameState,
        groups: Dict[str, List[Detection]],
        areas: Dict[str, Detection],
    ) -> None:
        cells = groups.get("cell", [])
        if not cells:
            return

        by_class: Dict[str, List[Detection]] = {}
        for d in cells:
            by_class.setdefault(d.class_name, []).append(d)

        # total_bet_cell and total_count_cell: assign via area containment.
        for cell_class, slot in (
            ("total_bet_cell", "total_bet"),
            ("total_count_cell", "total_count"),
        ):
            for cell in by_class.get(cell_class, []):
                bet_type = self._assign_cell_to_area(cell, areas)
                if bet_type is None:
                    continue
                text = self.ocr.read_text(self.detector.crop(frame, cell))
                bet = state.bets.setdefault(bet_type, BetState(bet_type=bet_type))
                setattr(bet, slot, text)

        # percent_cell: scoreboard rows, sort by y-coord and map by order.
        percent_cells = sorted(
            by_class.get("percent_cell", []), key=lambda d: d.center[1],
        )
        for row_idx, cell in enumerate(percent_cells):
            if row_idx >= len(self.percent_row_order):
                break
            bet_type = self.percent_row_order[row_idx]
            text = self.ocr.read_text(self.detector.crop(frame, cell))
            bet = state.bets.setdefault(bet_type, BetState(bet_type=bet_type))
            bet.percent = text

    @staticmethod
    def _assign_cell_to_area(
        cell: Detection, areas: Dict[str, Detection],
    ) -> Optional[str]:
        cx, cy = cell.center
        # 1) Containment: pick the smallest area whose bbox contains the cell.
        containing: List[Tuple[str, int]] = []
        for bet_type, area in areas.items():
            if area.contains_point(cx, cy):
                containing.append((bet_type, area.area_px))
        if containing:
            containing.sort(key=lambda x: x[1])
            return containing[0][0]

        # 2) Fallback: nearest area by center distance.
        if not areas:
            return None
        best = min(
            areas.items(),
            key=lambda kv: (kv[1].center[0] - cx) ** 2 + (kv[1].center[1] - cy) ** 2,
        )
        return best[0]

    @staticmethod
    def _infer_phase(state: GameState) -> str:
        if state.dice_result is not None:
            return "result"
        t = state.timer_int
        if t is not None and t > 0:
            return "betting"
        return "transition"


# Sanity check: the hard-coded lookups stay in sync with classes.CLASSES.
assert set(AREA_TO_BET.keys()).issubset(CLASS_NAME_TO_ID.keys()), \
    "AREA_TO_BET keys drifted from classes.py"
assert set(DICE_TO_OUTCOME.keys()).issubset(CLASS_NAME_TO_ID.keys()), \
    "DICE_TO_OUTCOME keys drifted from classes.py"


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/detect/xocdia/weights/best.pt")
    parser.add_argument("--source", required=True, help="Path to an image file.")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-annotated", default=None, help="Path to write annotated image.")
    return parser.parse_args()


def _main():
    import json
    import cv2

    args = _parse_args()
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.source}")

    pipe = GameAnalysisPipeline(
        weights=args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device,
    )
    state = pipe.analyze(frame)
    print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))

    if args.save_annotated:
        cv2.imwrite(args.save_annotated, pipe.annotate(frame, state))
        print(f"Annotated image saved to: {args.save_annotated}")


if __name__ == "__main__":
    _main()
