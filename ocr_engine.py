# ocr_engine.py - OCR engine
from paddleocr import PaddleOCR
import cv2
import numpy as np


class XocDiaOCR:
    def __init__(self):
        # The Xoc Dia game UI uses a stylised italic gold/cyan font for
        # money / count / percent values. PaddleOCR's auto-orientation
        # classifiers (``doc_orientation`` + ``textline_orientation``)
        # routinely mis-classify these glyphs as rotated 180 degrees
        # and silently flip the image - that is why we used to see
        # ``2.04M`` come back as ``W50'2`` (mirror of M=W, 5=S, .='),
        # ``5.02M`` as ``N2D'S``, ``31`` as ``IE``, etc.
        #
        # Our cells are already cropped horizontally LTR by YOLO, so
        # orientation classification has no signal it can usefully
        # contribute - it only hurts us. We disable all three
        # orientation steps in PaddleOCR 3.x and fall back to
        # ``use_angle_cls=False`` on the older 2.x API.
        v3_kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "lang": "en",
        }
        try:
            self.ocr = PaddleOCR(**v3_kwargs)
        except (TypeError, ValueError):
            v2_kwargs = {"use_angle_cls": False, "lang": "en"}
            try:
                self.ocr = PaddleOCR(**v2_kwargs, show_log=False)
            except (TypeError, ValueError):
                try:
                    self.ocr = PaddleOCR(**v2_kwargs)
                except ModuleNotFoundError as e:
                    if str(e).find("paddle") != -1:
                        raise ModuleNotFoundError(
                            "Missing dependency 'paddlepaddle'. "
                            "Install it in your venv, e.g. `pip install paddlepaddle`."
                        ) from e
                    raise

    def _run_ocr(self, image):
        """PaddleOCR 3.x: ocr() -> predict(); no 'cls' kwarg (use_textline_orientation in predict)."""
        try:
            return self.ocr.predict(image)
        except AttributeError:
            pass
        # Legacy 2.x: ocr(img, cls=True)
        try:
            return self.ocr.ocr(image, cls=True)
        except TypeError:
            return self.ocr.ocr(image)

    @staticmethod
    def _join_texts_from_result(result, conf_thresh=0.5):
        if result is None:
            return ""
        if not isinstance(result, (list, tuple)) or len(result) == 0:
            return ""

        first = result[0]

        # PaddleOCR 2.x: first page is list of lines [[box, (text, conf)], ...]
        if isinstance(first, list) and len(first) > 0:
            line0 = first[0]
            if (
                isinstance(line0, (list, tuple))
                and len(line0) >= 2
                and isinstance(line0[1], (list, tuple))
                and len(line0[1]) >= 2
            ):
                texts = []
                for line in first:
                    if not line or len(line) < 2:
                        continue
                    pair = line[1]
                    text, conf = pair[0], pair[1]
                    if conf > conf_thresh:
                        texts.append(text)
                return " ".join(texts)

        # PaddleOCR 3.x / PaddleX: OCRResult dict-like with rec_texts / rec_scores
        rec_texts = None
        rec_scores = None
        if isinstance(first, dict):
            rec_texts = first.get("rec_texts")
            rec_scores = first.get("rec_scores")
        else:
            if hasattr(first, "get"):
                rec_texts = first.get("rec_texts")
                rec_scores = first.get("rec_scores")
            if rec_texts is None and hasattr(first, "__getitem__"):
                try:
                    rec_texts = first["rec_texts"]
                except (KeyError, TypeError):
                    pass
                try:
                    rec_scores = first["rec_scores"]
                except (KeyError, TypeError):
                    pass

        if rec_texts:
            parts = []
            for i, t in enumerate(rec_texts):
                s = (
                    rec_scores[i]
                    if rec_scores is not None and i < len(rec_scores)
                    else 1.0
                )
                if s is None or float(s) >= conf_thresh:
                    parts.append(str(t) if not isinstance(t, tuple) else str(t[0]))
            return " ".join(parts)

        return ""

    def read_text(self, image, bbox=None):
        """
        Read text from image or specific region
        bbox: [x1, y1, x2, y2]
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            image = image[y1:y2, x1:x2]

        result = self._run_ocr(image)
        return self._join_texts_from_result(result)

    def read_number(self, image, bbox=None):
        """Read number specifically (for timer, roundId)"""
        text = self.read_text(image, bbox)

        import re

        numbers = re.findall(r"\d+", text)

        return numbers[0] if numbers else None
    