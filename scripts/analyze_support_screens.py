"""Analyze recent run screenshots with the project's own support vision.

Usage: python scripts/analyze_support_screens.py [limit]
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from fgo_bot.config import load_config  # noqa: E402
from fgo_bot.strategy import SupportSelector  # noqa: E402
from fgo_bot.vision import TemplateMatcher, read_image  # noqa: E402

CFG_PATH = PROJECT / "profiles" / "user.yaml"
RUNS = PROJECT / "runs"


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    cfg, _ = load_config(CFG_PATH)
    matcher = TemplateMatcher(cfg, CFG_PATH)
    selector = SupportSelector(cfg, CFG_PATH)

    images = sorted(
        RUNS.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in images[:limit]:
        if path.name.startswith("tmp_") or path.name.startswith("view"):
            continue
        img = read_image(path)
        scores = matcher.scores(img)
        support_score = next(
            (score for name, score in scores if name == "support_screen"),
            None,
        )
        readings = selector.support_level_readings(img)
        np5 = selector._np5_matches(img)
        try:
            guest_rows = selector.guest_support_row_ranges(
                img, respect_exclusion=False
            )
        except TypeError:
            guest_rows = selector.guest_support_row_ranges(img)
        has_normal = selector.has_normal_support_row(img)
        choice = selector.choose_highest_level_np(img)
        print(path.name)
        print(f"  support_screen: {support_score}")
        print(f"  guest_rows: {guest_rows}  has_normal: {has_normal}")
        print(
            "  readings: "
            + str(
                [
                    (r.value, r.maximum, round(r.confidence, 2), r.match.center)
                    for r in readings
                ]
            )
        )
        print(
            "  np5: "
            + str([(round(m.score, 2), m.center) for m in np5])
        )
        if choice is None:
            print("  choice: None")
        else:
            print(
                f"  choice: level={choice.candidate.get('level')} "
                f"np={choice.candidate.get('np_level')} "
                f"center={choice.match.center} score={choice.match.score:.2f}"
            )
        print()


if __name__ == "__main__":
    main()
