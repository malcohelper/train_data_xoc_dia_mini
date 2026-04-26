"""Post-process raw PaddleOCR output for the three text-cell classes.

The detector model is very accurate at *locating* cells (mAP50 ~ 0.99
on a healthy run), but PaddleOCR routinely confuses digits with
visually-similar letters (``8 <-> B``, ``0 <-> O``, ``1 <-> I/l``,
``5 <-> S``, ``2 <-> Z``...) and emits stray punctuation like ``'`` or
``,``. The realtime log used to surface that raw garbage straight to
the user (``total_bet=B.37M``, ``count=9OL``, ``PERCENT: ... 4_red 06
... 3w_1r 25M``).

This module:

1. Normalises common letter -> digit confusions per cell type.
2. Validates against a strict regex per cell type.
3. Returns ``None`` when the cleaned text doesn't look like a real
   value (callers can then render ``-`` instead of leaking junk into
   the logs).

All three sanitiser functions accept a single string and return either
the cleaned string or ``None``. They are intentionally pure / no
side-effects so they're trivially testable.

The class -> sanitiser mapping at the bottom is consumed by
``pipeline.py`` to wrap each ``ocr.read_text`` call.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional


# Letter -> digit confusables. PaddleOCR's English model in particular
# tends to flip digits to similarly-shaped letters when the source
# image is small / antialiased / has faint strokes (the entire game UI
# qualifies). We apply these substitutions BEFORE regex validation so a
# value like "B.37M" can recover to "8.37M" instead of being rejected
# wholesale.
#
# WARNING: do not include ``K`` / ``M`` here - they are legit suffixes
# for ``total_bet_cell``. Sanitisers that don't accept those suffixes
# strip them in their own pass.
_DIGIT_CONFUSABLES = {
    "B": "8",
    "b": "6",
    "O": "0",
    "o": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "L": "1",
    "|": "1",
    "!": "1",
    "Z": "2",
    "z": "2",
    "E": "3",
    "S": "5",
    "s": "5",
    "G": "6",
    "T": "7",
    "g": "9",
    "q": "9",
}


def _apply_confusables(text: str, allow_chars: str = "") -> str:
    """Substitute confusable letters with digits, except for letters
    listed in ``allow_chars`` (which are preserved as-is)."""
    out = []
    allow = set(allow_chars)
    for ch in text:
        if ch in allow:
            out.append(ch)
            continue
        out.append(_DIGIT_CONFUSABLES.get(ch, ch))
    return "".join(out)


# Strip ALL occurrences of whitespace and the noise-y punctuation
# PaddleOCR likes to inject (``'``, ``"``, ``,``, ``\``, ``/``). The
# game UI never uses these characters as legitimate separators inside
# a single cell, so removing them everywhere - not just at the
# boundaries - is intentional (e.g. ``"7 47M"`` -> ``"747M"``,
# ``"1,234"`` -> ``"1234"``).
_STRIP_NOISE = re.compile(r"[\s'\"`,\\/]+")


def _denoise(text: str) -> str:
    return _STRIP_NOISE.sub("", text or "")


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


# ---- per-class sanitisers ---------------------------------------------------


# Money values: 1-4 digits, optional decimal part, optional K/M suffix.
# Matches: ``45``, ``4266``, ``795K``, ``7.47M``, ``13.84M``.
_TOTAL_BET_RE = re.compile(r"^(\d{1,4})(?:\.(\d{1,3}))?([KkMm])?$")


def sanitise_total_bet(raw: Optional[str]) -> Optional[str]:
    """Clean a ``total_bet_cell`` OCR value.

    Returns the canonical form (uppercased K/M suffix) or ``None`` if
    the cleaned text doesn't match the expected pattern.
    """
    if not raw:
        return None
    text = _denoise(raw)
    # Reject text that has no original digits at all - prevents pure
    # letter junk like "Gi" / "WAIL" from being mapped to a fake
    # numeric value via confusables.
    if not _has_digit(text):
        return None
    # Allow K/M to survive confusable substitution.
    text = _apply_confusables(text, allow_chars="KkMm.")
    text = text.upper().rstrip(".")
    if not text:
        return None
    m = _TOTAL_BET_RE.match(text)
    if not m:
        return None
    intp, frac, suffix = m.groups()
    out = intp
    if frac:
        out += "." + frac
    if suffix:
        out += suffix.upper()
    return out


# Counts: pure non-negative integer, 1-4 digits typical (game caps in
# the hundreds; we leave headroom up to 9999).
_TOTAL_COUNT_RE = re.compile(r"^\d{1,4}$")


def sanitise_total_count(raw: Optional[str]) -> Optional[str]:
    """Clean a ``total_count_cell`` OCR value."""
    if not raw:
        return None
    text = _denoise(raw)
    # Same rationale as total_bet: don't fabricate a number out of
    # text that had no digits to begin with ("to" -> "0", "LE" -> "13",
    # etc.).
    if not _has_digit(text):
        return None
    text = _apply_confusables(text)
    # Drop anything that isn't a digit after substitution; OCR sometimes
    # picks up the trailing ``%`` from the scoreboard or stray dots.
    digits = re.sub(r"\D", "", text)
    if not digits or not _TOTAL_COUNT_RE.match(digits):
        return None
    return digits


# Percents: 1-3 digit number, optionally followed by a ``%``.
_PERCENT_NUM_RE = re.compile(r"\d{1,3}")


def sanitise_percent(raw: Optional[str]) -> Optional[str]:
    """Clean a ``percent_cell`` OCR value.

    The scoreboard cell is typically a tight crop around ``42%`` /
    ``58%`` etc., but if the bbox over-extends, OCR can return things
    like ``CHAN 47%`` or even part of a money amount. We extract the
    first 1-3 digit number, validate it's 0..100, and append ``%``.
    """
    if not raw:
        return None
    text = _denoise(raw)
    # Same rationale as total_bet / total_count: don't fabricate a
    # percent out of text that had no digits to begin with
    # ("LE" -> "13" -> "13%", "SO" -> "50%", "GO" -> "60%", etc.).
    if not _has_digit(text):
        return None
    # Drop literal ``%`` to make the regex simpler; we re-attach it.
    text = text.replace("%", "")
    text = _apply_confusables(text)
    m = _PERCENT_NUM_RE.search(text)
    if not m:
        return None
    val = int(m.group(0))
    if not (0 <= val <= 100):
        return None
    return f"{val}%"


# Mapping consumed by pipeline.py: { cell-class-name: sanitiser-fn }.
SANITISERS: Dict[str, Callable[[Optional[str]], Optional[str]]] = {
    "total_bet_cell": sanitise_total_bet,
    "total_count_cell": sanitise_total_count,
    "percent_cell": sanitise_percent,
}


def sanitise(class_name: str, raw: Optional[str]) -> Optional[str]:
    """Dispatch helper. Unknown class names pass through unchanged."""
    fn = SANITISERS.get(class_name)
    if fn is None:
        return raw
    return fn(raw)
