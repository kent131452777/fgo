"""Benchmark TemplateMatcher.find_first on real paused screenshots."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fgo_bot.config import load_config
from fgo_bot.vision import TemplateMatcher, read_image


def main() -> None:
    config, config_path = load_config(ROOT / "profiles" / "user.yaml")
    matcher = TemplateMatcher(config, config_path)
    frames = sorted((ROOT / "runs").glob("*.png"))
    if not frames:
        print("no frames")
        return
    frames = frames[:12]
    results = []
    for path in frames:
        screen = read_image(path)
        # warmup
        matcher.find_first(screen)
        start = time.perf_counter()
        repeated = 5
        for _ in range(repeated):
            match = matcher.find_first(screen)
        elapsed = (time.perf_counter() - start) / repeated
        label = (
            f"{match.rule['name']}@{match.score:.3f}"
            if match is not None
            else "None"
        )
        results.append((path.name, elapsed, label))
        print(f"{path.name}: {elapsed * 1000:.1f} ms  ->  {label}")
    total = sum(item[1] for item in results)
    print(f"avg {total / len(results) * 1000:.1f} ms/frame")


if __name__ == "__main__":
    main()
