"""Single source of truth for the 17-class Xoc Dia detection schema.

Keep this file in sync with ``xocdia.yaml`` at all times. Everything else
(label_tool, detector, eval, pipeline) imports from here so there is no
risk of drift.
"""

# Ordered class mapping. Class IDs MUST match xocdia.yaml.
CLASSES = {
    0: "round_id",
    1: "timer",
    2: "new_round",
    3: "area_chan",
    4: "area_le",
    5: "area_4_red",
    6: "area_3w_1r",
    7: "area_3r_1w",
    8: "area_4_white",
    9: "dice_4r",
    10: "dice_4w",
    11: "dice_3w1r",
    12: "dice_3r1w",
    13: "dice_2w2r",
    14: "percent_cell",
    15: "total_bet_cell",
    16: "total_count_cell",
}

# BGR colors tuned to roughly match each element's in-game appearance.
COLORS = {
    0: (0, 0, 220),
    1: (0, 220, 255),
    2: (60, 180, 255),
    3: (0, 165, 255),
    4: (0, 220, 0),
    5: (255, 255, 0),
    6: (255, 128, 0),
    7: (180, 80, 200),
    8: (180, 0, 180),
    9: (40, 40, 255),
    10: (240, 240, 240),
    11: (120, 120, 255),
    12: (120, 80, 255),
    13: (180, 180, 80),
    14: (0, 255, 180),
    15: (50, 220, 255),
    16: (220, 220, 220),
}

# Semantic grouping used by the UI (label tool status bar) and the detector
# (to split detections by category before downstream processing).
CLASS_GROUPS = [
    ("state", [0, 1, 2]),
    ("area", [3, 4, 5, 6, 7, 8]),
    ("dice", [9, 10, 11, 12, 13]),
    ("cell", [14, 15, 16]),
]

# Expected number of instances per frame. Used by check_labels.py to flag
# frames that look under- or over-labeled. "None" means "0 or 1", i.e.
# the element only appears during certain game phases.
EXPECTED_INSTANCES_PER_FRAME = {
    0: (0, 1),     # round_id
    1: (0, 1),     # timer (appears during countdown only)
    2: (0, 1),     # new_round banner
    3: (1, 1),     # area_chan
    4: (1, 1),     # area_le
    5: (1, 1),     # area_4_red
    6: (1, 1),     # area_3w_1r
    7: (1, 1),     # area_3r_1w
    8: (1, 1),     # area_4_white
    9: (0, 1),     # dice_* (exactly one visible when bowl opens)
    10: (0, 1),
    11: (0, 1),
    12: (0, 1),
    13: (0, 1),
    14: (0, 6),    # percent_cell (6 when scoreboard visible)
    15: (0, 6),    # total_bet_cell
    16: (0, 6),    # total_count_cell
}


# Reverse lookup: class_name -> class_id.
CLASS_NAME_TO_ID = {name: cid for cid, name in CLASSES.items()}


def category_of(class_id: int) -> str:
    """Return the semantic category ('state'/'area'/'dice'/'cell')."""
    for cat, ids in CLASS_GROUPS:
        if class_id in ids:
            return cat
    return "unknown"
