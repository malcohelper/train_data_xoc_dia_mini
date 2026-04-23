# realtime_capture.py - Capture và detect real-time
import mss
import cv2
import numpy as np
from detector import XocDiaDetector
import time
from pathlib import Path


def resolve_model_path(model_path, fallback_pattern):
    path = Path(model_path)
    if path.exists():
        return str(path)

    candidates = sorted(Path(".").glob(fallback_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return model_path

class RealtimeCapture:
    def __init__(self, model_path='best.pt', sub_model_path=None):
        resolved_stage1 = resolve_model_path(model_path, "runs/**/weights/best.pt")
        resolved_stage2 = None
        if sub_model_path:
            resolved_stage2 = resolve_model_path(sub_model_path, "runs/**/sub*/weights/best.pt")

        print(f"📦 Stage 1 model: {resolved_stage1}")
        if resolved_stage2:
            print(f"📦 Stage 2 model: {resolved_stage2}")
        else:
            print("📦 Stage 2 model: not found, running stage 1 only")

        self.detector = XocDiaDetector(model_path=resolved_stage1, sub_model_path=resolved_stage2)
        self.sct = mss.mss()

        # Default capture region (can be changed live with ROI selector)
        self.monitor = {
            "top": 0,
            "left": 0,
            "width": 1280,
            "height": 800,
        }

        self.last_round_id = None
        self.preview_window = "XocDia Preview"
        self.last_result = None
        self.running = True

    def start(self, interval=2, show_preview=True):
        """
        Start real-time detection
        interval: seconds between captures
        """
        print("🎮 Starting real-time detection...")
        print("⌨️  Hotkeys: r=select game region | s=save current frame | q=quit")

        last_detect_ts = 0.0

        while self.running:
            try:
                screenshot = self.capture()
                now = time.time()

                if now - last_detect_ts >= interval:
                    cv2.imwrite("temp_capture.png", screenshot)
                    result = self.detector.detect_image(screenshot)
                    self.last_result = result
                    last_detect_ts = now
                    self.handle_result(result, screenshot)

                if show_preview:
                    self.show_preview(screenshot)
                    key = cv2.waitKey(1) & 0xFF
                    self.handle_hotkeys(key, screenshot)

                time.sleep(0.03)
            except KeyboardInterrupt:
                print("\n👋 Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(0.2)

        if show_preview:
            cv2.destroyAllWindows()

    def handle_result(self, result, screenshot):
        """Print and persist only when a new round is detected."""
        if result["roundId"] and result["roundId"] != self.last_round_id:
            print("\n" + "=" * 60)
            print(f"🎲 NEW ROUND: {result['roundId']}")
            print(f"⏱️  Timer: {result['timer']}s")
            print(f"🏆 Winner: {result['winner']}")
            print(f"🎯 Dice: {result['diceCount']['red']} đỏ, {result['diceCount']['white']} trắng")
            print(f"📊 Result: {result['result']}")
            for area_name, area_info in result.get("betAreas", {}).items():
                if area_info.get("isActive"):
                    print(
                        f"🧩 {area_name}: count={area_info.get('sub_count')} | "
                        f"money={area_info.get('sub_money')}"
                    )
            print("=" * 60 + "\n")

            self.last_round_id = result["roundId"]
            self.save_round(result, screenshot)

    def show_preview(self, screenshot):
        """Render preview window to help align capture region."""
        preview = screenshot.copy()
        h, w = preview.shape[:2]

        cv2.putText(
            preview,
            f"Region: left={self.monitor['left']} top={self.monitor['top']} w={self.monitor['width']} h={self.monitor['height']}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "Hotkeys: r=select region | s=save frame | q=quit",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        if self.last_result:
            self.draw_detection_overlay(preview, self.last_result)
            info = (
                f"Round={self.last_result.get('roundId')} "
                f"Timer={self.last_result.get('timer')} "
                f"Winner={self.last_result.get('winner')} "
                f"Pass={self.last_result.get('detectionPass', 'primary')}"
            )
            cv2.putText(
                preview,
                info,
                (10, max(70, h - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        cv2.imshow(self.preview_window, preview)

    def draw_detection_overlay(self, frame, result):
        """Draw detected boxes and labels onto preview frame."""
        colors = {
            "round_id": (255, 255, 0),
            "timer": (0, 255, 255),
            "new_round": (80, 200, 255),
            "area_chan": (0, 165, 255),
            "area_le": (0, 255, 0),
            "area_4_red": (255, 255, 0),
            "area_3w_1r": (255, 0, 0),
            "area_3r_1w": (255, 0, 255),
            "area_4_white": (128, 0, 128),
            "4r": (0, 0, 255),
            "4w": (220, 220, 220),
            "3w1r": (255, 80, 80),
            "3r1w": (255, 120, 200),
            "2w2r": (180, 180, 0),
            "sub_count": (255, 255, 255),
            "sub_money": (0, 215, 255),
        }

        regions = result.get("regions", {})
        for key in ("round_id", "timer", "new_round"):
            item = regions.get(key, {})
            bbox = item.get("bbox")
            if not bbox:
                continue
            label = key
            if key == "round_id" and result.get("roundId"):
                label = f"round_id: {result['roundId']}"
            if key == "timer" and result.get("timer") is not None:
                label = f"timer: {result['timer']}"
            if key == "new_round":
                label = f"new_round: {result.get('isNewRound')}"
            self._draw_box_with_label(frame, bbox, label, colors[key], thickness=2)

        for area_name, area_info in result.get("betAreas", {}).items():
            bbox = area_info.get("bbox")
            if not bbox:
                continue

            area_label = area_name
            if area_info.get("sub_count") is not None or area_info.get("sub_money") is not None:
                area_label += f" | c={area_info.get('sub_count')} m={area_info.get('sub_money')}"

            color = colors.get(area_name, (200, 200, 200))
            thickness = 3 if result.get("detectedArea") == area_name else 2
            self._draw_box_with_label(frame, bbox, area_label, color, thickness=thickness)

            for sub_box in area_info.get("sub_boxes", []):
                sub_bbox = sub_box.get("bbox")
                sub_class = sub_box.get("class")
                if not sub_bbox or not sub_class:
                    continue
                self._draw_box_with_label(
                    frame,
                    sub_bbox,
                    sub_class,
                    colors.get(sub_class, (180, 180, 180)),
                    thickness=1,
                )

    def _draw_box_with_label(self, frame, bbox, label, color, thickness=2):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        text_y = y1 - 8 if y1 > 20 else y1 + 18
        cv2.putText(
            frame,
            str(label),
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    def handle_hotkeys(self, key, screenshot):
        if key == ord("q"):
            self.running = False
        elif key == ord("r"):
            self.select_region_with_mouse()
        elif key == ord("s"):
            cv2.imwrite("preview_capture.png", screenshot)
            print("💾 Saved current preview frame -> preview_capture.png")

    def select_region_with_mouse(self):
        """
        Press hotkey 'r' to select game area by drag-and-drop once.
        """
        full_monitor = self.sct.monitors[1]
        full_img = np.array(self.sct.grab(full_monitor))
        full_img = cv2.cvtColor(full_img, cv2.COLOR_BGRA2BGR)

        print("🖱️  Drag to select game region, then press ENTER/SPACE. ESC to cancel.")
        x, y, w, h = cv2.selectROI("Select Game Region", full_img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select Game Region")

        if w > 0 and h > 0:
            self.monitor = {
                "top": int(full_monitor["top"] + y),
                "left": int(full_monitor["left"] + x),
                "width": int(w),
                "height": int(h),
            }
            print(f"✅ Updated region: {self.monitor}")
        else:
            print("ℹ️  Region selection cancelled.")

    def capture(self):
        """Capture screen"""
        img = np.array(self.sct.grab(self.monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    
    def save_round(self, result, screenshot):
        """Save round data"""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        round_id = result['roundId'].replace('#', '')
        Path("rounds").mkdir(exist_ok=True)
        
        # Save image
        cv2.imwrite(f'rounds/{round_id}_{timestamp}.png', screenshot)
        
        # Save JSON
        with open(f'rounds/{round_id}_{timestamp}.json', 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

# Run
if __name__ == "__main__":
    capture = RealtimeCapture(
        model_path='runs/detect/train/weights/best.pt',
        sub_model_path='runs/detect/sub_train/weights/best.pt'
    )
    capture.start(interval=2, show_preview=True)  # Check mỗi 2s