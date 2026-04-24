"""Single source of truth for the 15-class Xoc Dia detection schema.

Keep this file in sync with ``xocdia.yaml`` at all times. Everything else
(label_tool, detector, eval, pipeline) imports from here so there is no
risk of drift.

Schema changes in this revision:
    - Dropped class ``round_id`` (timestamp is used instead).
    - Dropped class ``new_round`` (timer >= 46 transition signals start).
"""

# Ordered class mapping. Class IDs MUST match xocdia.yaml.
CLASSES = {
    0: "timer",
    1: "area_chan",
    2: "area_le",
    3: "area_4_red",
    4: "area_3w_1r",
    5: "area_3r_1w",
    6: "area_4_white",
    7: "dice_4r",
    8: "dice_4w",
    9: "dice_3w1r",
    10: "dice_3r1w",
    11: "dice_2w2r",
    12: "percent_cell",
    13: "total_bet_cell",
    14: "total_count_cell",
}

# BGR colors tuned to roughly match each element's in-game appearance.
COLORS = {
    0: (0, 220, 255),      # timer - yellow/amber
    1: (0, 165, 255),      # area_chan - orange
    2: (0, 220, 0),        # area_le - green
    3: (255, 255, 0),      # area_4_red - cyan
    4: (255, 128, 0),      # area_3w_1r - blue
    5: (180, 80, 200),     # area_3r_1w - pink
    6: (180, 0, 180),      # area_4_white - purple
    7: (40, 40, 255),      # dice_4r - red
    8: (240, 240, 240),    # dice_4w - white
    9: (120, 120, 255),    # dice_3w1r
    10: (120, 80, 255),    # dice_3r1w
    11: (180, 180, 80),    # dice_2w2r
    12: (0, 255, 180),     # percent_cell
    13: (50, 220, 255),    # total_bet_cell
    14: (220, 220, 220),   # total_count_cell
}

# Semantic grouping used by the UI (label tool status bar) and the detector
# (to split detections by category before downstream processing).
CLASS_GROUPS = [
    ("state", [0]),
    ("area", [1, 2, 3, 4, 5, 6]),
    ("dice", [7, 8, 9, 10, 11]),
    ("cell", [12, 13, 14]),
]

# Expected number of instances per frame. Used by check_labels.py to flag
# frames that look under- or over-labeled. "(0, 1)" = "0 or 1", i.e. the
# element only appears during certain game phases.
EXPECTED_INSTANCES_PER_FRAME = {
    0: (0, 1),     # timer (appears during countdown only)
    1: (1, 1),     # area_chan
    2: (1, 1),     # area_le
    3: (1, 1),     # area_4_red
    4: (1, 1),     # area_3w_1r
    5: (1, 1),     # area_3r_1w
    6: (1, 1),     # area_4_white
    7: (0, 1),     # dice_* (exactly one visible when bowl opens)
    8: (0, 1),
    9: (0, 1),
    10: (0, 1),
    11: (0, 1),
    12: (0, 6),    # percent_cell (6 when scoreboard visible)
    13: (0, 6),    # total_bet_cell
    14: (0, 6),    # total_count_cell
}


# Reverse lookup: class_name -> class_id.
CLASS_NAME_TO_ID = {name: cid for cid, name in CLASSES.items()}


def category_of(class_id: int) -> str:
    """Return the semantic category ('state'/'area'/'dice'/'cell')."""
    for cat, ids in CLASS_GROUPS:
        if class_id in ids:
            return cat
    return "unknown"
