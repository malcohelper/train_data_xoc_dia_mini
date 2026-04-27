"""Per-cell-class image preprocessing for OCR.

The detector reaches mAP50 ~ 0.99, so cell crops are tight - but the raw
crops still feed PaddleOCR with stylised game-UI text on coloured
backgrounds (yellow money on cyan, white counts on cyan, tiny black
percents on white scoreboard). PaddleOCR's CRNN was trained on
photographic English text and confuses ``B<->8 / O<->0 / I<->1 / S<->5``
on this domain at distressing rates.

For each of the three text-cell classes we apply a tiny, deterministic
pipeline that turns the crop into a clean **black-on-white, upscaled,
padded** binary image - which is what the recognition CRNN handles
best:

1. Colour mask the foreground (HSV yellow / brightness threshold) when
   the foreground colour is consistent. Falls back to plain Otsu on the
   value channel if the mask is too sparse.
2. Otsu binarisation, with auto-invert based on which polarity has more
   border pixels (so output is always black text on white background).
3. Bicubic upscale to a target text height (~64px for body text, ~80px
   for the smaller scoreboard percents). PaddleOCR's CRNN expects ~32px
   text; going larger gives it more pixels to disambiguate stylised
   strokes without changing aspect ratio.
4. Constant white-padding so the network sees clear margins.

The three preprocessors are pure / no-side-effects and dispatched
through the ``PREPROCESSORS`` map (consumed by ``pipeline.py``).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import cv2
import numpy as np


# Target text height post-upscale. CRNN sweet spot is ~32px; we go
# larger to give the recogniser more pixels for stylised strokes.
_TARGET_TEXT_HEIGHT = 64
# Percents are tiny in source (16-20px) so we upscale further.
_PERCENT_TEXT_HEIGHT = 80
# White margin around the digit so the recogniser sees clear borders.
_PAD = 8
_PAD_PERCENT = 12

# Yellow / orange-yellow money text. HSV ranges are wide on purpose -
# the game uses gradient strokes plus shadow which spreads saturation
# and value. False positives outside the cell don't matter because the
# crop is already tight.
_YELLOW_LO = (12, 80, 100)
_YELLOW_HI = (40, 255, 255)

# Brightness threshold for white count digits on cyan background.
# Counts are rendered close to pure white (V ~ 230+); 180 keeps a safety
# margin for anti-aliased edges.
_WHITE_GRAY_THRESH = 180

# Mask is only "useful" if at least this fraction of pixels are
# foreground; otherwise fall back to plain Otsu.
_MIN_MASK_FILL = 0.02


def _safe(crop: Optional[np.ndarray]) -> bool:
    return (
        crop is not None
        and getattr(crop, "size", 0) > 0
        and crop.ndim >= 2
        and crop.shape[0] >= 4
        and crop.shape[1] >= 4
    )


def _upscale(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = max(1.0, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _pad_white(img: np.ndarray, border: int = _PAD) -> np.ndarray:
    if img.ndim == 2:
        return cv2.copyMakeBorder(
            img, border, border, border, border,
            cv2.BORDER_CONSTANT, value=255,
        )
    return cv2.copyMakeBorder(
        img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _otsu_black_on_white(gray: np.ndarray) -> np.ndarray:
    """Otsu-binarise ``gray`` and ensure the result is *black text on
    white background* (auto-inverted based on border majority)."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = th.shape
    border = np.concatenate(
        [th[0, :], th[-1, :], th[:, 0], th[:, -1]]
    )
    if border.mean() < 127:
        th = cv2.bitwise_not(th)
    return th


def _mask_fill_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def prep_total_bet(crop: np.ndarray) -> np.ndarray:
    """Yellow money digits -> black-on-white, upscaled, padded."""
    if not _safe(crop):
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _YELLOW_LO, _YELLOW_HI)
    if _mask_fill_ratio(mask) < _MIN_MASK_FILL:
        # No yellow detected (atypical money colour or empty cell):
        # fall back to plain Otsu on the value channel so we still try.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bw = _otsu_black_on_white(gray)
    else:
        # Mask: digit pixels are 255 (white), background is 0 (black).
        # Invert to get the canonical black-on-white form OCR likes.
        bw = cv2.bitwise_not(mask)
    bw = _upscale(bw, _TARGET_TEXT_HEIGHT)
    bw = _pad_white(bw, _PAD)
    return _to_bgr(bw)


def prep_total_count(crop: np.ndarray) -> np.ndarray:
    """White count digits on cyan/dark -> black-on-white."""
    if not _safe(crop):
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, _WHITE_GRAY_THRESH, 255, cv2.THRESH_BINARY)
    if _mask_fill_ratio(mask) < _MIN_MASK_FILL:
        bw = _otsu_black_on_white(gray)
    else:
        bw = cv2.bitwise_not(mask)
    bw = _upscale(bw, _TARGET_TEXT_HEIGHT)
    bw = _pad_white(bw, _PAD)
    return _to_bgr(bw)


def prep_percent(crop: np.ndarray) -> np.ndarray:
    """Small black text on white scoreboard -> upscaled grayscale.
    We deliberately do *not* binarise here: the percents are 16-20px
    tall in source, so thin strokes in glyphs like ``9`` and ``%`` get
    eaten by Otsu and confuse the recogniser. CLAHE-boosted grayscale
    preserves the strokes while still giving the recogniser high local
    contrast."""
    if not _safe(crop):
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = _upscale(gray, _PERCENT_TEXT_HEIGHT)
    gray = _pad_white(gray, _PAD_PERCENT)
    return _to_bgr(gray)


PREPROCESSORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "total_bet_cell": prep_total_bet,
    "total_count_cell": prep_total_count,
    "percent_cell": prep_percent,
}


def preprocess(cell_class: str, crop: np.ndarray) -> np.ndarray:
    """Dispatch helper. Returns the original crop unchanged for class
    names not in ``PREPROCESSORS`` (e.g. ``timer``)."""
    fn = PREPROCESSORS.get(cell_class)
    if fn is None:
        return crop
    out = fn(crop)
    return out if out is not None else crop
