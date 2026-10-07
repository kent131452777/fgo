"""Print per-row components and chosen parse for a support screen."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from fgo_bot.config import load_config  # noqa: E402
from fgo_bot.strategy import SupportSelector  # noqa: E402
from fgo_bot.vision import read_image  # noqa: E402

CFG_PATH = PROJECT / "profiles" / "user.yaml"
RUNS = PROJECT / "runs"


def main() -> None:
    cfg, _ = load_config(CFG_PATH)
    selector = SupportSelector(cfg, CFG_PATH)
    for name in sys.argv[1:] or ["20260823-160617-178-PAUSED.png"]:
        img = read_image(RUNS / name)
        rows = selector.normal_support_row_ranges(img)
        print(f"\n########## {name} rows={rows} ##########")
        scale_x = img.shape[1] / selector.base_width
        scale_y = img.shape[0] / selector.base_height
        for top, bottom in rows:
            raw_roi = selector.settings.get(
                "support_level_value_roi", [130, 180, 270, 890]
            )
            x1 = round(raw_roi[0] * scale_x)
            x2 = round(raw_roi[2] * scale_x)
            off = selector.settings.get("support_level_row_text_offsets", [7, 45])
            y1 = top + round(off[0] * scale_y)
            y2 = top + round(off[1] * scale_y)
            comps = selector._support_level_components(img, x1, y1, x2, y2)
            reading = selector._parse_level_sequence(img, comps)
            print(
                f"row [{top},{bottom}] strip y[{y1},{y2}]: "
                f"{[(c[0], c[1], c[2], c[3], c[4], round(c[5], 2)) for c in comps]}"
            )
            print(
                "  -> "
                + (str(reading) if reading is not None else "None")
            )


if __name__ == "__main__":
    main()
