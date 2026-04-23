# detector.py - Two-stage detection pipeline
from ultralytics import YOLO
from ocr_engine import XocDiaOCR
import cv2
import numpy as np
from pathlib import Path


class XocDiaDetector:
    def __init__(self, model_path="best.pt", sub_model_path=None, conf=0.35, sub_conf=0.35, fallback_conf=0.2):
        # Stage 1 model: detect large regions
        self.model = YOLO(model_path)
        self.conf = conf
        self.fallback_conf = fallback_conf

        # Stage 2 model: detect sub_count/sub_money inside each area (optional)
        self.sub_model = None
        if sub_model_path and Path(sub_model_path).exists():
            self.sub_model = YOLO(sub_model_path)
        self.sub_conf = sub_conf

        # OCR
        self.ocr = XocDiaOCR()

        # Stage 1 classes
        self.classes = {
            0: "round_id",
            1: "timer",
            2: "area_chan",
            3: "area_le",
            4: "area_4_red",
            5: "area_3w_1r",
            6: "area_3r_1w",
            7: "area_4_white",
            8: "new_round",
            9: "4r",
            10: "4w",
            11: "3w1r",
            12: "3r1w",
            13: "2w2r",
        }

        # Stage 2 classes
        self.sub_classes = {0: "sub_count", 1: "sub_money"}

        self.side_area_classes = (
            "area_chan",
            "area_le",
        )
        self.result_area_classes = (
            "area_4_red",
            "area_3w_1r",
            "area_3r_1w",
            "area_4_white",
            "4r",
            "3w1r",
            "3r1w",
            "4w",
            "2w2r",
        )
        self.region_classes = self.side_area_classes + self.result_area_classes

    def detect(self, image_path):
        """Detect full game state from screenshot path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect_image(image)

    def detect_image(self, image):
        """Detect full game state directly from numpy image."""
        detections = self._extract_detections(image, conf=self.conf)
        result = self.calculate_result(detections)
        result["detectionPass"] = "primary"

        if self._needs_fallback(result):
            enhanced = self._enhance_for_detection(image)
            detections_fb = self._extract_detections(enhanced, conf=self.fallback_conf)
            result_fb = self.calculate_result(detections_fb)
            result_fb["detectionPass"] = "fallback"

            if self._score_result(result_fb) >= self._score_result(result):
                return result_fb

        return result

    def _extract_detections(self, image, conf):
        results = self.model(image, conf=conf)[0]
        detections = {
            "round_id_candidates": [],
            "timer_candidates": [],
            "new_round_candidates": [],
        }
        for area_name in self.region_classes:
            detections[area_name] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.classes.get(cls_id)
            if cls_name is None:
                continue

            bbox = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = self._sanitize_bbox(bbox, image.shape)
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if cls_name == "round_id":
                detections["round_id_candidates"].append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "text": self.ocr.read_text(roi),
                    }
                )
            elif cls_name == "timer":
                number = self.ocr.read_number(roi)
                detections["timer_candidates"].append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "value": int(number) if number else None,
                    }
                )
            elif cls_name == "new_round":
                detections["new_round_candidates"].append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                    }
                )
            else:
                area_det = {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "brightness": self.calculate_brightness(roi),
                    "isActive": self.is_box_active(roi),
                }
                if self.sub_model is not None:
                    area_det.update(self.detect_sub_fields(image, [x1, y1, x2, y2]))
                detections[cls_name].append(area_det)

        return detections

    def _enhance_for_detection(self, image):
        """Light enhancement for hard frames (dark/compressed/noisy)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        merged = cv2.merge((h, s, v))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
        return enhanced

    def _needs_fallback(self, result):
        if result.get("detectedArea"):
            return False
        if result.get("roundId") and result.get("timer") is not None:
            return False
        return True

    def _score_result(self, result):
        score = 0.0
        if result.get("roundId"):
            score += 2.0
        if result.get("timer") is not None:
            score += 1.5
        if result.get("detectedArea"):
            score += 2.0
        active_regions = 0
        for area in result.get("betAreas", {}).values():
            if area.get("bbox") is not None:
                active_regions += 1
        score += 0.2 * active_regions
        return score

    def detect_sub_fields(self, image, area_bbox):
        """Stage 2 detection inside one area: sub_count and sub_money."""
        x1, y1, x2, y2 = self._sanitize_bbox(area_bbox, image.shape)
        area_roi = image[y1:y2, x1:x2]
        if area_roi.size == 0:
            return {"sub_count": None, "sub_money": None, "sub_boxes": []}

        sub_result = self.sub_model(area_roi, conf=self.sub_conf)[0]
        sub_boxes = []
        best_count = None
        best_money = None

        for box in sub_result.boxes:
            sub_cls_id = int(box.cls[0])
            sub_cls_name = self.sub_classes.get(sub_cls_id)
            if sub_cls_name is None:
                continue

            local_bbox = box.xyxy[0].cpu().numpy().astype(int)
            sx1, sy1, sx2, sy2 = self._sanitize_bbox(local_bbox, area_roi.shape)
            sub_roi = area_roi[sy1:sy2, sx1:sx2]
            if sub_roi.size == 0:
                continue

            conf = float(box.conf[0])
            global_bbox = [x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2]

            if sub_cls_name == "sub_count":
                value = self.ocr.read_number(sub_roi)
                if value and (best_count is None or conf > best_count["conf"]):
                    best_count = {"value": int(value), "text": value, "conf": conf}
            else:
                text = self.ocr.read_text(sub_roi)
                if text and (best_money is None or conf > best_money["conf"]):
                    best_money = {"value": text, "conf": conf}

            sub_boxes.append({"class": sub_cls_name, "bbox": global_bbox, "conf": conf})

        return {
            "sub_count": best_count["value"] if best_count else None,
            "sub_money": best_money["value"] if best_money else None,
            "sub_boxes": sub_boxes,
        }

    def calculate_brightness(self, roi):
        """Calculate average brightness."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def is_box_active(self, roi):
        """Check if box is active (bright)."""
        return self.calculate_brightness(roi) > 100

    def _best_detection(self, items):
        if not items:
            return None
        return max(items, key=lambda x: x["conf"])

    def _sanitize_bbox(self, bbox, shape):
        h, w = shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return x1, y1, x2, y2

    def calculate_result(self, detections):
        """Convert raw detections to final output schema."""
        area_to_dice = {
            "area_4_red": (4, 0),
            "area_3r_1w": (3, 1),
            "area_3w_1r": (1, 3),
            "area_4_white": (0, 4),
            "4r": (4, 0),
            "3r1w": (3, 1),
            "3w1r": (1, 3),
            "4w": (0, 4),
            "2w2r": (2, 2),
        }

        area_scores = {}
        for area_name in self.result_area_classes:
            best = self._best_detection(detections[area_name])
            area_scores[area_name] = best["conf"] if best else 0.0

        selected_area = max(area_scores, key=area_scores.get) if area_scores else None
        red_count, white_count = (0, 0)
        if selected_area and area_scores[selected_area] > 0:
            red_count, white_count = area_to_dice[selected_area]

        chan_best = self._best_detection(detections["area_chan"])
        le_best = self._best_detection(detections["area_le"])
        chan_active = chan_best["isActive"] if chan_best else False
        le_active = le_best["isActive"] if le_best else False
        round_best = self._best_detection(detections["round_id_candidates"])
        timer_best = self._best_detection(detections["timer_candidates"])
        new_round_best = self._best_detection(detections["new_round_candidates"])

        area_details = {}
        for area_name in self.region_classes:
            best = self._best_detection(detections[area_name])
            area_details[area_name] = {
                "bbox": best["bbox"] if best else None,
                "conf": best["conf"] if best else 0.0,
                "isActive": best["isActive"] if best else False,
                "sub_count": best.get("sub_count") if best else None,
                "sub_money": best.get("sub_money") if best else None,
                "sub_boxes": best.get("sub_boxes", []) if best else [],
            }

        return {
            "roundId": round_best["text"] if round_best else None,
            "timer": timer_best["value"] if timer_best else None,
            "isNewRound": bool(new_round_best),
            "winner": "CHAN" if chan_active else ("LE" if le_active else None),
            "diceCount": {
                "red": red_count,
                "white": white_count,
                "total": red_count + white_count,
            },
            "result": "LE" if red_count > white_count else ("CHAN" if red_count < white_count else "DRAW"),
            "detectedArea": selected_area,
            "regions": {
                "round_id": {
                    "bbox": round_best["bbox"] if round_best else None,
                    "conf": round_best["conf"] if round_best else 0.0,
                },
                "timer": {
                    "bbox": timer_best["bbox"] if timer_best else None,
                    "conf": timer_best["conf"] if timer_best else 0.0,
                },
                "new_round": {
                    "bbox": new_round_best["bbox"] if new_round_best else None,
                    "conf": new_round_best["conf"] if new_round_best else 0.0,
                },
            },
            "betAreas": area_details,
        }


if __name__ == "__main__":
    detector = XocDiaDetector(
        model_path="runs/detect/train/weights/best.pt",
        sub_model_path="runs/detect/sub_train/weights/best.pt",
    )

    result = detector.detect("screenshot.png")

    print("=" * 50)
    print(f"Round ID: {result['roundId']}")
    print(f"Timer: {result['timer']}s")
    print(f"Winner: {result['winner']}")
    print(f"Dice: {result['diceCount']['red']} đỏ, {result['diceCount']['white']} trắng")
    print(f"Result: {result['result']}")
    print(f"Detected area: {result['detectedArea']}")
    print("=" * 50)