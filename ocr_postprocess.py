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
    "J": "1",  # PP-OCRv5 reads stylised italic ``1.45M`` as ``J.45M``.
    "j": "1",
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


# Money values: 1-5 digits, optional decimal part, optional K/M suffix.
# Matches: ``45``, ``4266``, ``795K``, ``7.47M``, ``13.84M`` and the
# K-misread shapes we recover below (``7674`` -> ``767K``,
# ``32814`` -> ``328K``).
_TOTAL_BET_RE = re.compile(r"^(\d{1,5})(?:\.(\d{1,3}))?([KkMm])?$")


def sanitise_total_bet(raw: Optional[str]) -> Optional[str]:
    """Clean a ``total_bet_cell`` OCR value.

    Returns the canonical form (uppercased K/M suffix) or ``None`` if
    the cleaned text doesn't match the expected pattern.
    """
    if not raw:
        return None
    # Decimal-comma recovery: the game UI uses ``.`` for the decimal
    # mark, but PaddleOCR's English model occasionally reads the
    # stylised italic dot as ``,`` (giving ``6,03M`` instead of
    # ``6.03M``). Convert before ``_denoise`` strips the comma.
    text = raw.replace(",", ".")
    text = _denoise(text)
    # Either an original digit OR a K/M suffix is required. Without an
    # anchor we'd fabricate a value out of pure-letter junk like ``LE``
    # via confusables. ``IK`` / ``TM`` style readings ARE accepted -
    # the K/M suffix itself is a strong "this is a money cell" anchor
    # and we want to recover ``1K`` / ``7M`` from those.
    if not (_has_digit(text) or any(ch in "KkMm" for ch in text)):
        return None
    # Allow K/M to survive confusable substitution.
    text = _apply_confusables(text, allow_chars="KkMm.")
    text = text.upper().rstrip(".")
    if not text:
        return None
    # Trailing K-misread recovery: a stylised italic capital ``K``
    # sometimes comes back as ``K3`` / ``K4`` (the K's lower-right
    # serif is glued to a neighbour glyph and read as a digit). E.g.
    # ``857K`` -> ``857K4``, ``117.1K`` (period dropped) -> ``1171K3``.
    # Strip the stray trailing 3/4 BEFORE the regex match so the
    # value validates as ``\d+K``.
    text = re.sub(r"(\d)K[34]$", r"\1K", text)
    # Trailing M-misread recovery: PaddleOCR occasionally reads the
    # italic ``M`` suffix as ``N`` (the leaning diagonals of an italic
    # M can lose one stroke at low resolution, e.g. ``3.06M`` -> ``3.06N``).
    # ``N`` is not a valid character in any money-cell shape, so a
    # trailing standalone ``N`` is almost certainly a misread ``M``.
    text = re.sub(r"N$", "M", text)
    m = _TOTAL_BET_RE.match(text)
    if not m:
        return None
    intp, frac, suffix = m.groups()
    # The regex was widened to ``\d{1,5}`` only to make the
    # ``K=14`` recovery branch below reachable. A 5-digit integer
    # part is not a value the game UI can show (anything >= 1000
    # always renders as a fractional like ``12.34M``, never as
    # ``12345`` / ``12345K`` / ``12345.6``), so reject it up front
    # unless it's a candidate for K=14 recovery (5 digits ending
    # in ``14`` with no suffix).
    if len(intp) == 5 and not (suffix is None and frac is None and intp.endswith("14")):
        return None
    out = intp
    if frac:
        out += "." + frac
    if suffix:
        # ``text`` was uppercased before the regex match, so the suffix
        # is already canonical - no extra .upper() needed.
        out += suffix
    else:
        # PaddleOCR confusable: the trailing ``K`` of a money value
        # like ``767K`` is sometimes recognised as a digit. We've
        # observed two flavours of this misread on the live game:
        #   1) ``K`` -> ``4``  (e.g. ``767K`` -> ``7674``)
        #   2) ``K`` -> ``14`` (e.g. ``328K`` -> ``32814``)
        # The game UI always uses a ``K``/``M`` suffix for values
        # >= 1000, so a plain 4-digit integer ending in ``4`` or a
        # 5-digit integer ending in ``14`` is much more likely to be a
        # misread ``XXXK`` than a legit 4/5-digit bet. Recover both.
        # This may false-positive on legit values like ``1234`` /
        # ``32114`` (rare in this game).
        if frac is None and len(out) == 5 and out.endswith("14"):
            out = out[:-2] + "K"
        elif frac is None and len(out) == 4 and out.endswith("4"):
            out = out[:-1] + "K"
    return out


# Counts: pure non-negative integer, 1-4 digits typical (game caps in
# the hundreds; we leave headroom up to 9999).
_TOTAL_COUNT_RE = re.compile(r"^\d{1,4}$")


def sanitise_total_count(raw: Optional[str]) -> Optional[str]:
    """Clean a ``total_count_cell`` OCR value."""
    if not raw:
        return None
    text = _denoise(raw)
    # Counts have no suffix to anchor against, so the rule is weaker
    # than total_bet's: accept if the text has an original digit, OR
    # at least 2 confusable letters (``III`` -> ``111``, ``IS`` ->
    # ``15``, ``ISI`` -> ``151`` are real recoveries from the live
    # game). The 2-char minimum stops single-character junk like ``s``
    # / ``l`` from being fabricated into ``5`` / ``1``.
    confusable_count = sum(1 for ch in text if ch in _DIGIT_CONFUSABLES)
    if not (_has_digit(text) or confusable_count >= 2):
        return None
    # NB: counts apply confusables BEFORE stripping non-digits (the
    # opposite of sanitise_percent). The asymmetry is deliberate -
    # counts can be 1-4 digits with no upper sanity bound (the regex
    # accepts any value 0..9999), so an over-confused result like
    # "9OL" -> "901" is still bounded by the digit-count regex and
    # is much more likely to be a true positive recovery than a false
    # positive. Percents are constrained 0..100, so an over-confused
    # "58b" -> "586" trips the bound and silently rejects a valid
    # read - hence percents prefer strip-then-(no-confuse) over
    # confuse-then-strip.
    text = _apply_confusables(text)
    digits = re.sub(r"\D", "", text)
    if not digits or not _TOTAL_COUNT_RE.match(digits):
        return None
    return digits


# Percents: 1-3 digit number, optionally followed by a ``%``.
_PERCENT_NUM_RE = re.compile(r"\d{1,3}")


_PERCENT_KEEP_SLASH_RE = re.compile(r"[\s'\"`,\\]+")


def sanitise_percent(raw: Optional[str]) -> Optional[str]:
    """Clean a ``percent_cell`` OCR value.

    The scoreboard cell is typically a tight crop around ``42%`` /
    ``58%`` etc., but if the bbox over-extends, OCR can return things
    like ``CHAN 47%`` or even part of a money amount. We extract the
    first 1-3 digit number, validate it's 0..100, and append ``%``.

    Three-stage match strategy (each later stage is a fallback for
    when the prior one fails to produce a valid 0..100 value):

    1. ``%``-anchored: ``(\\d{1,3})\\s*%`` against the raw text. This
       cleanly handles trailing OCR junk like ``12%6`` / ``40%6`` by
       only taking the digits before the ``%``.
    2. Fully denoised: strip noise (incl. ``/``) and take the first
       digit run. Recovers values like ``1/0`` -> ``10%`` (where the
       slash was a stray artefact between the ``1`` and ``0``).
    3. Slash-as-boundary: strip noise but KEEP ``/``, then take the
       first digit run. Recovers ``79/0`` -> ``79%`` (where ``%`` was
       mis-rendered as ``/0`` and we want ``/`` to act as a digit-run
       boundary, not be silently dropped).
    """
    if not raw:
        return None
    # Stage 1: ``%``-anchored.
    m = re.search(r"(\d{1,3})\s*%", raw)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return f"{val}%"
    # Stage 2: fully denoised (slash stripped).
    text_b = _denoise(raw).replace("%", "")
    m = _PERCENT_NUM_RE.search(text_b)
    if m:
        val = int(m.group(0))
        if 0 <= val <= 100:
            return f"{val}%"
    # Stage 3: slash kept as digit boundary.
    text_a = _PERCENT_KEEP_SLASH_RE.sub("", raw).replace("%", "")
    m = _PERCENT_NUM_RE.search(text_a)
    if m:
        val = int(m.group(0))
        if 0 <= val <= 100:
            return f"{val}%"
    return None


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
