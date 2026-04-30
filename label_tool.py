import argparse
import copy
from pathlib import Path

import cv2

from classes import CLASSES, CLASS_GROUPS, COLORS


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def resolve_class_filter(tokens):
    """Turn --filter-classes tokens into a set of class IDs.

    Accepts any mix of:
    - numeric class ID (e.g. ``7``)
    - class name from ``CLASSES`` (e.g. ``dice_4r``)
    - category name from ``CLASS_GROUPS`` (``state`` / ``area`` / ``dice``
      / ``cell``), which expands to every class ID in that category.
    """
    name_to_id = {name: cid for cid, name in CLASSES.items()}
    cat_to_ids = {cat: set(ids) for cat, ids in CLASS_GROUPS}
    ids: set = set()
    for raw in tokens:
        tok = raw.strip()
        if not tok:
            continue
        if tok.isdigit():
            cid = int(tok)
            if cid not in CLASSES:
                raise SystemExit(
                    f"Unknown class id {cid!r} in --filter-classes. "
                    f"Valid: {sorted(CLASSES)}."
                )
            ids.add(cid)
        elif tok in name_to_id:
            ids.add(name_to_id[tok])
        elif tok in cat_to_ids:
            ids.update(cat_to_ids[tok])
        else:
            raise SystemExit(
                f"Unknown --filter-classes token: {tok!r}. Valid: class id, "
                f"class name ({sorted(name_to_id)}), or category "
                f"({sorted(cat_to_ids)})."
            )
    return ids


# Pretty labels used only for the status bar (keeps the existing UI wording).
_STATUS_GROUP_LABELS = {
    "state": "State",
    "area": "Areas",
    "dice": "Dice",
    "cell": "Cells",
}
_STATUS_GROUPS = [
    (_STATUS_GROUP_LABELS.get(name, name.title()), ids)
    for name, ids in CLASS_GROUPS
]


class LabelTool:
    def __init__(
        self,
        images_folder="dataset/images/train",
        labels_folder="dataset/labels/train",
        only_unlabeled=False,
        autosave=True,
        auto_next_class=True,
        filter_class_ids=None,
        detector_weights=None,
        detector_conf=0.25,
        detector_imgsz=800,
        detector_imgsz_fallback=1280,
    ):
        self.images_folder = Path(images_folder)
        self.labels_folder = Path(labels_folder)
        self.labels_folder.mkdir(parents=True, exist_ok=True)

        self.classes = CLASSES
        self.colors = COLORS

        self.only_unlabeled = only_unlabeled
        self.autosave = autosave
        self.auto_next_class = auto_next_class
        # ``filter_class_ids`` is None for no filter, or a non-empty set
        # of class ids - in which case ``collect_images`` is restricted
        # to frames whose label file contains at least one of those ids.
        self.filter_class_ids = set(filter_class_ids) if filter_class_ids else None

        # On-demand YOLO auto-detect (bound to the 'y' hotkey). The
        # detector is lazy-loaded on first use so startup stays fast
        # when the user never presses 'y'.
        self.detector_weights = detector_weights
        self.detector_conf = detector_conf
        self.detector_imgsz = detector_imgsz
        self.detector_imgsz_fallback = detector_imgsz_fallback
        self.detector = None
        self.current_class = 0
        self.boxes = []
        self.drawing = False
        self.start_point = None
        self.temp_point = None
        self.window_name = "Label Tool"
        self.current_img = None
        self.current_img_path = None
        self.current_index = 0

        # Text-input class picker state. Active while the user is typing a
        # 1-2 digit class number after pressing '/'.
        self.class_input_mode = False
        self.class_input_buffer = ""

        # Index of the currently selected box (None = no selection).
        self.selected_box = None

        # Template slots (0-9). Each slot is None or a list of box dicts.
        self.templates = {i: None for i in range(10)}
        # Template mode: None, "save", or "apply".
        self.template_mode = None

        self.images = self.collect_images()
        if self.filter_class_ids:
            total_before = len(self.images)
            self.images = [
                p for p in self.images
                if self._label_has_any_class(p, self.filter_class_ids)
            ]
            names = ", ".join(
                sorted(self.classes[c] for c in self.filter_class_ids)
            )
            print(
                f"[filter] Showing {len(self.images)}/{total_before} images "
                f"whose labels contain any of: {names}"
            )
        if self.only_unlabeled:
            self.images = [p for p in self.images if not self.get_label_path(p).exists()]
        else:
            self.current_index = self._find_last_labeled_index()

    def collect_images(self):
        images = []
        for ext in IMAGE_EXTS:
            images.extend(self.images_folder.glob(f"*{ext}"))
        return sorted(images)

    def _find_last_labeled_index(self):
        """Return index of the last image that already has a label file."""
        last = 0
        for i, img in enumerate(self.images):
            if self.get_label_path(img).exists():
                last = i
        return last

    def get_label_path(self, img_path: Path):
        return self.labels_folder / f"{img_path.stem}.txt"

    def _label_has_any_class(self, img_path: Path, wanted: set) -> bool:
        """Return True if ``img_path``'s label file has a box whose class
        id is in ``wanted``. Missing/empty label files count as no match.
        """
        lbl = self.get_label_path(img_path)
        if not lbl.exists():
            return False
        try:
            lines = lbl.read_text().splitlines()
        except OSError:
            return False
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid in wanted:
                return True
        return False

    def load_labels(self, label_path: Path):
        loaded = []
        if not label_path.exists():
            return loaded
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in self.classes:
                continue
            loaded.append({"class": cls_id, "bbox": [float(x) for x in parts[1:]]})
        return loaded

    def save_labels(self, label_path: Path):
        lines = []
        for box in self.boxes:
            line = f"{box['class']} " + " ".join(f"{x:.6f}" for x in box["bbox"])
            lines.append(line)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"Saved: {label_path.name} ({len(self.boxes)} boxes)")

    def copy_from_previous(self):
        if self.current_index <= 0:
            print("No previous image to copy from.")
            return
        prev_label = self.get_label_path(self.images[self.current_index - 1])
        if not prev_label.exists():
            print("Previous image has no label.")
            return
        self.boxes = self.load_labels(prev_label)
        print(f"Copied {len(self.boxes)} boxes from {prev_label.name}")

    def _ensure_detector(self):
        """Lazy-load the YOLO detector used by the 'y' hotkey."""
        if self.detector is not None:
            return
        if not self.detector_weights:
            print(
                "[auto-detect] --weights was not provided; 'y' hotkey is "
                "disabled. Pass --weights <path/to/best.pt> on startup."
            )
            return
        from detector import XocDiaDetector

        print(f"[auto-detect] Loading weights: {self.detector_weights}")
        self.detector = XocDiaDetector(
            weights=self.detector_weights,
            conf=self.detector_conf,
            imgsz=self.detector_imgsz,
            imgsz_fallback=self.detector_imgsz_fallback,
        )
        print(
            f"[auto-detect] Ready. conf={self.detector_conf} "
            f"imgsz={self.detector_imgsz}+{self.detector_imgsz_fallback}. "
            f"Press 'y' on any frame to (re)run detection."
        )

    @staticmethod
    def _yolo_iou(a, b):
        """IoU between two YOLO-format boxes [cx, cy, w, h] (normalised)."""
        ax1 = a[0] - a[2] / 2.0
        ay1 = a[1] - a[3] / 2.0
        ax2 = a[0] + a[2] / 2.0
        ay2 = a[1] + a[3] / 2.0
        bx1 = b[0] - b[2] / 2.0
        by1 = b[1] - b[3] / 2.0
        bx2 = b[0] + b[2] / 2.0
        by2 = b[1] + b[3] / 2.0
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def auto_detect_current_frame(self, replace=False, iou_thresh=0.45):
        """Run the YOLO detector on the current image and MERGE its
        predictions into the canvas (default) or REPLACE all existing
        boxes (when ``replace=True``).

        Merge rule: a predicted box is appended only if it does NOT
        overlap an existing box of the SAME class with IoU >= iou_thresh.
        This preserves any manually-drawn boxes while filling in classes
        the user has not yet labelled.
        """
        self._ensure_detector()
        if self.detector is None or self.current_img is None:
            return

        h, w = self.current_img.shape[:2]
        if h <= 0 or w <= 0:
            return

        dets = self.detector.detect(self.current_img)
        pred_boxes = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            # Clamp to the image bounds (detector already does this but
            # belt-and-braces for YOLO format safety).
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            pred_boxes.append({"class": int(d.class_id), "bbox": [cx, cy, bw, bh]})

        prev_count = len(self.boxes)

        if replace:
            self.boxes = pred_boxes
            self.selected_box = None
            print(
                f"[auto-detect] {self.current_img_path.name}: REPLACED "
                f"{prev_count} existing -> {len(pred_boxes)} predictions"
            )
            return

        # MERGE: append predictions that don't overlap an existing box
        # of the same class (IoU < threshold).
        added = 0
        skipped = 0
        for p in pred_boxes:
            dup = False
            for existing in self.boxes:
                if existing["class"] != p["class"]:
                    continue
                if self._yolo_iou(existing["bbox"], p["bbox"]) >= iou_thresh:
                    dup = True
                    break
            if dup:
                skipped += 1
            else:
                self.boxes.append(p)
                added += 1
        self.selected_box = None
        print(
            f"[auto-detect] {self.current_img_path.name}: merged "
            f"{added} new (skipped {skipped} dupes of existing) -> "
            f"{len(self.boxes)} total boxes"
        )

    def _find_box_at(self, x, y):
        """Return index of the smallest box containing (x, y), or None."""
        if self.current_img is None:
            return None
        h, w = self.current_img.shape[:2]
        best_idx, best_area = None, float("inf")
        for i, box in enumerate(self.boxes):
            cx, cy, bw, bh = box["bbox"]
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = bw * bh
                if area < best_area:
                    best_idx, best_area = i, area
        return best_idx

    def _try_select_box(self, x, y):
        """Toggle selection of the box at (x, y)."""
        idx = self._find_box_at(x, y)
        if idx is not None:
            if self.selected_box == idx:
                self.selected_box = None
                print("Deselected box")
            else:
                self.selected_box = idx
                box = self.boxes[idx]
                print(f"Selected box {idx}: {self.classes[box['class']]}")
        else:
            self.selected_box = None

    def mouse_callback(self, event, x, y, _flags, _param):
        if self.current_img is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.temp_point = (x, y)
            dx = abs(self.temp_point[0] - self.start_point[0])
            dy = abs(self.temp_point[1] - self.start_point[1])
            if dx < 5 and dy < 5:
                self._try_select_box(x, y)
            else:
                self.add_box_from_points(self.start_point, self.temp_point)
            self.start_point = None
            self.temp_point = None

    def add_box_from_points(self, p1, p2):
        if self.current_img is None:
            return
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            return

        h, w = self.current_img.shape[:2]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = abs(x2 - x1) / w
        height = abs(y2 - y1) / h

        self.boxes.append(
            {
                "class": self.current_class,
                "bbox": [x_center, y_center, width, height],
            }
        )
        print(f"Added: {self.classes[self.current_class]} | total={len(self.boxes)}")
        if self.auto_next_class:
            self.current_class = (self.current_class + 1) % len(self.classes)
            print(f"Auto-next class -> {self.current_class}:{self.classes[self.current_class]}")

    def set_class(self, cls_id):
        if cls_id in self.classes:
            self.current_class = cls_id
            print(f"Class -> {cls_id}:{self.classes[cls_id]}")
        else:
            print(f"Invalid class id: {cls_id}")

    def draw_boxes(self, frame):
        h, w = frame.shape[:2]
        for i, box in enumerate(self.boxes):
            cx, cy, bw, bh = box["bbox"]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cls_id = box["class"]
            is_selected = (i == self.selected_box)
            color = self.colors.get(cls_id, (200, 200, 200))
            thickness = 3 if cls_id == self.current_class else 2

            if is_selected:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), thickness + 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"[SEL] {self.classes[cls_id]}"
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = self.classes[cls_id]

            cv2.putText(
                frame,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255) if is_selected else color,
                2,
            )

        if self.drawing and self.start_point and self.temp_point:
            cv2.rectangle(frame, self.start_point, self.temp_point, (255, 255, 255), 1)

    def draw_status(self, frame):
        idx = self.current_index + 1
        total = len(self.images)
        title = (
            f"[{idx}/{total}] {self.current_img_path.name} | "
            f"class={self.current_class}:{self.classes[self.current_class]}"
        )
        cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        controls_1 = "0-9: class | j/k: prev/next | / + NN + Enter: jump class | t: auto-next"
        controls_2 = "u: undo | x: clear class | c: copy prev | Click: select | d: del sel | Esc: desel"
        controls_3 = "w+0-9: save tpl | e+0-9: apply tpl | p: prev | space: next | s: save | a: autosave | q: quit"
        y_hint = "y: YOLO auto-detect (merge) | Y: replace all" if self.detector_weights else ""
        cv2.putText(frame, controls_1, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        cv2.putText(frame, controls_2, (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        cv2.putText(frame, controls_3, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        if y_hint:
            cv2.putText(frame, y_hint, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)

        sel_text = ""
        if self.selected_box is not None and self.selected_box < len(self.boxes):
            sb = self.boxes[self.selected_box]
            sel_text = f" | selected=[{self.selected_box}] {self.classes[sb['class']]}"

        status = (
            f"autosave={'ON' if self.autosave else 'OFF'} | "
            f"auto-next-class={'ON' if self.auto_next_class else 'OFF'} | "
            f"boxes={len(self.boxes)}{sel_text}"
        )
        status_y = 120 if self.detector_weights else 102
        cv2.putText(frame, status, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 80), 1)

        # Mode banners
        extra_y = 0
        banner_y_base = 142 if self.detector_weights else 124
        if self.class_input_mode:
            cv2.putText(
                frame,
                f"Class input: '{self.class_input_buffer}_' (Enter to apply, Esc to cancel)",
                (10, banner_y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                2,
            )
            extra_y += 22

        if self.template_mode is not None:
            action = "SAVE to" if self.template_mode == "save" else "APPLY from"
            used = [str(i) for i in range(10) if self.templates[i] is not None]
            slots_str = ",".join(used) if used else "none"
            cv2.putText(
                frame,
                f"Template {action} slot 0-9 (used: [{slots_str}]) | Esc to cancel",
                (10, banner_y_base + extra_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 180),
                2,
            )
            extra_y += 22

        # Per-group class counts grid (4 lines).
        class_counts = {k: 0 for k in self.classes}
        for box in self.boxes:
            class_counts[box["class"]] += 1

        y = banner_y_base + extra_y
        for group_name, ids in _STATUS_GROUPS:
            parts = [f"{cid}:{self.classes[cid]}({class_counts[cid]})" for cid in ids]
            line = f"{group_name:<6}| " + "  ".join(parts)
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
            )
            y += 18

    def _apply_class_input(self):
        if not self.class_input_buffer:
            print("Empty class input, cancelled.")
        else:
            try:
                cls_id = int(self.class_input_buffer)
                self.set_class(cls_id)
            except ValueError:
                print(f"Invalid class input: {self.class_input_buffer}")
        self.class_input_mode = False
        self.class_input_buffer = ""

    def handle_class_input_key(self, key):
        """Handle keys while in class-input mode. Returns True if consumed."""
        if key == 13 or key == 10:  # Enter
            self._apply_class_input()
            return True
        if key == 27:  # Esc
            print("Class input cancelled.")
            self.class_input_mode = False
            self.class_input_buffer = ""
            return True
        if key == 8 or key == 127:  # Backspace / Delete
            self.class_input_buffer = self.class_input_buffer[:-1]
            return True
        if ord("0") <= key <= ord("9") and len(self.class_input_buffer) < 2:
            self.class_input_buffer += chr(key)
            # Auto-apply when exactly 2 digits entered.
            if len(self.class_input_buffer) == 2:
                self._apply_class_input()
            return True
        return True  # consume all other keys silently while typing

    def handle_template_key(self, key):
        """Handle keys while in template save/apply mode. Returns True if consumed."""
        if key == 27:  # Esc
            print("Template mode cancelled.")
            self.template_mode = None
            return True
        if ord("0") <= key <= ord("9"):
            slot = key - ord("0")
            if self.template_mode == "save":
                self.templates[slot] = copy.deepcopy(self.boxes)
                n = len(self.boxes)
                print(f"Template saved: slot {slot} ({n} boxes)")
            elif self.template_mode == "apply":
                if self.templates[slot] is None:
                    print(f"Template slot {slot} is empty")
                else:
                    self.boxes = copy.deepcopy(self.templates[slot])
                    self.selected_box = None
                    print(f"Template applied: slot {slot} ({len(self.boxes)} boxes)")
            self.template_mode = None
            return True
        return True  # consume other keys silently

    def handle_key(self, key):
        label_path = self.get_label_path(self.current_img_path)

        if self.template_mode is not None:
            self.handle_template_key(key)
            return "stay"

        if self.class_input_mode:
            self.handle_class_input_key(key)
            return "stay"

        if key == ord("/"):
            self.class_input_mode = True
            self.class_input_buffer = ""
            print("Class input mode. Type 0-16 then press Enter (Esc to cancel).")
            return "stay"

        if key == 27:  # Esc
            if self.selected_box is not None:
                self.selected_box = None
                print("Deselected box")
            return "stay"

        if key == ord("d"):
            if self.selected_box is not None:
                removed = self.boxes.pop(self.selected_box)
                print(f"Deleted box {self.selected_box}: {self.classes[removed['class']]}")
                self.selected_box = None
            else:
                print("No box selected (right-click a box first)")
            return "stay"

        if ord("0") <= key <= ord("9"):
            self.set_class(key - ord("0"))
            return "stay"

        if key == ord("j"):
            self.set_class((self.current_class - 1) % len(self.classes))
            return "stay"

        if key == ord("k"):
            self.set_class((self.current_class + 1) % len(self.classes))
            return "stay"

        if key == ord("u") and self.boxes:
            if self.selected_box is not None and self.selected_box >= len(self.boxes) - 1:
                self.selected_box = None
            self.boxes.pop()
            print("Undo last box")
            return "stay"

        if key == ord("x"):
            before = len(self.boxes)
            self.boxes = [b for b in self.boxes if b["class"] != self.current_class]
            self.selected_box = None
            print(f"Cleared class {self.current_class}: removed {before - len(self.boxes)} boxes")
            return "stay"

        if key == ord("c"):
            self.copy_from_previous()
            self.selected_box = None
            return "stay"

        if key == ord("a"):
            self.autosave = not self.autosave
            print(f"Autosave -> {'ON' if self.autosave else 'OFF'}")
            return "stay"

        if key == ord("t"):
            self.auto_next_class = not self.auto_next_class
            print(f"Auto-next class -> {'ON' if self.auto_next_class else 'OFF'}")
            return "stay"

        if key == ord("w"):
            self.template_mode = "save"
            print("Template SAVE mode. Press 0-9 to pick slot (Esc to cancel).")
            return "stay"

        if key == ord("e"):
            self.template_mode = "apply"
            print("Template APPLY mode. Press 0-9 to pick slot (Esc to cancel).")
            return "stay"

        if key == ord("s"):
            self.save_labels(label_path)
            return "stay"

        if key == ord("y"):
            self.auto_detect_current_frame(replace=False)
            return "stay"

        if key == ord("Y"):
            self.auto_detect_current_frame(replace=True)
            return "stay"

        if key == ord("p"):
            if self.autosave:
                self.save_labels(label_path)
            return "prev"

        if key == ord(" ") or key == ord("n"):
            if self.autosave:
                self.save_labels(label_path)
            return "next"

        if key == ord("q"):
            if self.autosave:
                self.save_labels(label_path)
            return "quit"

        return "stay"

    def run_single_image(self, img_path: Path):
        self.current_img_path = img_path
        self.current_img = cv2.imread(str(img_path))
        if self.current_img is None:
            print(f"Skip unreadable image: {img_path}")
            return "next"

        self.boxes = self.load_labels(self.get_label_path(img_path))
        self.selected_box = None
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            display = self.current_img.copy()
            self.draw_boxes(display)
            self.draw_status(display)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(16) & 0xFF
            action = self.handle_key(key)
            if action in {"next", "prev", "quit"}:
                return action

    def label_images(self):
        if not self.images:
            print(f"No images found in {self.images_folder}")
            return

        print(f"Images folder: {self.images_folder}")
        print(f"Labels folder: {self.labels_folder}")
        print(f"Total images: {len(self.images)}")
        print(f"Classes: {len(self.classes)}")
        if self.current_index > 0:
            print(f"Resuming at image {self.current_index + 1}/{len(self.images)}: {self.images[self.current_index].name}")

        while 0 <= self.current_index < len(self.images):
            action = self.run_single_image(self.images[self.current_index])
            if action == "next":
                self.current_index += 1
            elif action == "prev":
                self.current_index = max(0, self.current_index - 1)
            elif action == "quit":
                break

        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Fast YOLO label tool for train/val.")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset root folder.")
    parser.add_argument("--split", choices=["train", "val"], default="train", help="Label split.")
    parser.add_argument("--images-folder", default=None, help="Custom images folder.")
    parser.add_argument("--labels-folder", default=None, help="Custom labels folder.")
    parser.add_argument(
        "--only-unlabeled",
        action="store_true",
        help="Open only images that do not have label file yet.",
    )
    parser.add_argument(
        "--no-autosave",
        action="store_true",
        help="Disable autosave when changing image.",
    )
    parser.add_argument(
        "--no-auto-next-class",
        action="store_true",
        help="Disable auto jump to next class after each drawn box.",
    )
    parser.add_argument(
        "--filter-classes",
        nargs="+",
        default=None,
        metavar="ID|NAME|CATEGORY",
        help="Only open images whose label file contains at least one "
             "box matching any of the given tokens. Accepts class ids "
             "(e.g. 7), class names (e.g. dice_4r), or category names "
             "(state/area/dice/cell, expanded to all ids in the group). "
             "Example: --filter-classes dice  -> every frame with any "
             "dice_* detection. Useful for QA'ing a single class group "
             "without copying files into a separate folder.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a YOLO best.pt. When provided, enables the 'y' "
             "hotkey which runs the detector on the current frame and "
             "replaces the canvas boxes with its predictions (you can "
             "then edit / delete them like normal).",
    )
    parser.add_argument(
        "--auto-detect-conf",
        type=float,
        default=0.25,
        help="Min confidence for boxes returned by 'y' auto-detect. "
             "Lower = more predictions (noisier but more recall). "
             "Default 0.25.",
    )
    parser.add_argument(
        "--auto-detect-imgsz",
        type=int,
        default=800,
        help="Primary imgsz for 'y' auto-detect (default 800).",
    )
    parser.add_argument(
        "--auto-detect-imgsz-fallback",
        type=int,
        default=1280,
        help="Secondary imgsz (multi-scale ensemble) for 'y' "
             "auto-detect (default 1280, 0 to disable).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    images_folder = args.images_folder or f"{args.dataset_root}/images/{args.split}"
    labels_folder = args.labels_folder or f"{args.dataset_root}/labels/{args.split}"

    filter_class_ids = (
        resolve_class_filter(args.filter_classes) if args.filter_classes else None
    )

    tool = LabelTool(
        images_folder=images_folder,
        labels_folder=labels_folder,
        only_unlabeled=args.only_unlabeled,
        autosave=not args.no_autosave,
        auto_next_class=not args.no_auto_next_class,
        filter_class_ids=filter_class_ids,
        detector_weights=args.weights,
        detector_conf=args.auto_detect_conf,
        detector_imgsz=args.auto_detect_imgsz,
        detector_imgsz_fallback=args.auto_detect_imgsz_fallback,
    )
    tool.label_images()
