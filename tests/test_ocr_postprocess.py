"""Unit tests for ``ocr_postprocess``.

These cover the recovery cases observed in live-game ``debug_cells/``
dumps. Run with ``python -m pytest tests/`` (no other deps needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_postprocess import (
    sanitise_percent,
    sanitise_total_bet,
    sanitise_total_count,
)


# ---- total_bet ---------------------------------------------------------


def test_total_bet_passes_canonical_values():
    assert sanitise_total_bet("1.45M") == "1.45M"
    assert sanitise_total_bet("767K") == "767K"
    assert sanitise_total_bet("45") == "45"
    assert sanitise_total_bet("13.84M") == "13.84M"


def test_total_bet_decimal_comma_recovers():
    # PaddleOCR mis-reads the stylised italic decimal dot as a comma.
    assert sanitise_total_bet("6,03M") == "6.03M"
    assert sanitise_total_bet("1,68M") == "1.68M"
    assert sanitise_total_bet("7,29M") == "7.29M"
    assert sanitise_total_bet("1,42M") == "1.42M"


def test_total_bet_letter_only_with_suffix_recovers():
    # ``IK`` / ``TM`` etc. - no original digit but a clear K/M anchor.
    assert sanitise_total_bet("IK") == "1K"
    assert sanitise_total_bet("IIK") == "11K"
    assert sanitise_total_bet("IM") == "1M"
    assert sanitise_total_bet("TM") == "7M"
    assert sanitise_total_bet("IJM") == "11M"
    assert sanitise_total_bet("J.ISM") == "1.15M"
    assert sanitise_total_bet("I.BIM") == "1.81M"


def test_total_bet_letter_only_without_suffix_rejects():
    # Pure-letter junk with no K/M anchor must NOT fabricate a value.
    assert sanitise_total_bet("LE") is None
    assert sanitise_total_bet("LL") is None
    assert sanitise_total_bet("Hi") is None
    assert sanitise_total_bet("") is None
    assert sanitise_total_bet("}]") is None


def test_total_bet_k_misread_recovery():
    # Trailing italic capital ``K`` mis-read as ``K3`` / ``K4``.
    assert sanitise_total_bet("857k4") == "857K"
    assert sanitise_total_bet("35k4") == "35K"
    assert sanitise_total_bet("1k4") == "1K"
    assert sanitise_total_bet("1171K3") == "1171K"
    assert sanitise_total_bet("662k4") == "662K"


def test_total_bet_legacy_k_recovery_still_works():
    # 4-digit ending in 4 -> XXXK, 5-digit ending in 14 -> XXXK
    assert sanitise_total_bet("7674") == "767K"
    assert sanitise_total_bet("32814") == "328K"


def test_total_bet_5digit_plain_int_rejects():
    # 5-digit values without a K/M-recovery candidate are bogus.
    assert sanitise_total_bet("12345") is None
    assert sanitise_total_bet("12345.6") is None
    assert sanitise_total_bet("12345K") is None


# ---- total_count -------------------------------------------------------


def test_total_count_passes_normal_values():
    assert sanitise_total_count("42") == "42"
    assert sanitise_total_count("0") == "0"
    assert sanitise_total_count("999") == "999"
    assert sanitise_total_count("9OL") == "901"  # confusables


def test_total_count_letter_only_recovers_when_2plus_confusables():
    assert sanitise_total_count("II") == "11"
    assert sanitise_total_count("III") == "111"
    assert sanitise_total_count("IS") == "15"
    assert sanitise_total_count("ISI") == "151"


def test_total_count_single_letter_or_junk_rejects():
    assert sanitise_total_count("") is None
    assert sanitise_total_count("s") is None  # only 1 confusable
    assert sanitise_total_count("}]") is None
    assert sanitise_total_count("}}") is None
    assert sanitise_total_count("•") is None


# ---- percent -----------------------------------------------------------


def test_percent_passes_normal_values():
    assert sanitise_percent("42%") == "42%"
    assert sanitise_percent("100%") == "100%"
    assert sanitise_percent("0%") == "0%"
    assert sanitise_percent("CHAN 47%") == "47%"


def test_percent_trailing_junk_after_percent_recovers():
    # PaddleOCR sometimes reads the trailing pixel noise as an extra
    # digit AFTER the ``%``.
    assert sanitise_percent("12%6") == "12%"
    assert sanitise_percent("40%6") == "40%"


def test_percent_slash_as_percent_glyph_recovers():
    # The ``%`` glyph is occasionally rendered as ``/0``; with the
    # ``/`` acting as a digit-run boundary we still get the correct
    # value.
    assert sanitise_percent("79/0") == "79%"
    assert sanitise_percent("38/0") == "38%"


def test_percent_slash_as_inner_artefact_recovers():
    # ``1/0`` is a legit ``10%`` reading (slash is artefact between
    # the ``1`` and ``0``), not a slash-as-glyph case. Stage 2
    # (full denoise) handles this before stage 3 sees it.
    assert sanitise_percent("1/0") == "10%"


def test_percent_no_digits_rejects():
    assert sanitise_percent("") is None
    assert sanitise_percent("CAN") is None
    assert sanitise_percent("LE") is None
    assert sanitise_percent("te") is None


def test_percent_out_of_range_rejects():
    assert sanitise_percent("250%") is None
    assert sanitise_percent("999") is None
