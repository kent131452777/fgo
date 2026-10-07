"""Verify pyramid matching agrees with the exact full-resolution path."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fgo_bot.vision as vision
from fgo_bot.config import load_config
from fgo_bot.vision import TemplateMatcher, read_image


def main() -> None:
    config, config_path = load_config(ROOT / "profiles" / "user.yaml")
    matcher = TemplateMatcher(config, config_path)
    frames = sorted((ROOT / "runs").glob("*.png"))[:8]

    # 全分辨率精确路径：把金字塔最小面积调大即可强制走旧路径。
    exact = TemplateMatcher(config, config_path)
    mismatches = 0
    checked = 0
    for path in frames:
        screen = read_image(path)
        for rule in config["rules"]:
            if not rule.get("enabled", True) or rule.get("detect_only"):
                continue
            if rule["name"] not in matcher._templates:
                continue
            fast = matcher.match_rule(screen, rule, {})
            vision.PYRAMID_MIN_AREA = 10 ** 9
            slow = exact.match_rule(screen, rule, {})
            vision.PYRAMID_MIN_AREA = 200_000
            checked += 1
            if (fast is None) != (slow is None):
                mismatches += 1
                print(f"DECISION DIFF {path.name} {rule['name']}: "
                      f"fast={fast and fast.score} slow={slow and slow.score}")
                continue
            if fast is None:
                continue
            dscore = abs(fast.score - slow.score)
            dist = abs(fast.x - slow.x) + abs(fast.y - slow.y)
            if dscore > 1e-6 or dist > 8:
                mismatches += 1
                print(f"VALUE DIFF {path.name} {rule['name']}: "
                      f"fast=({fast.score:.6f},{fast.x},{fast.y}) "
                      f"slow=({slow.score:.6f},{slow.x},{slow.y})")
    print(f"checked {checked} rule/frame pairs, mismatches: {mismatches}")


if __name__ == "__main__":
    main()
