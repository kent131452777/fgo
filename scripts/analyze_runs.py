# -*- coding: utf-8 -*-
"""批量分析 runs 目录下截图：模板得分 + 技能弹窗/选人界面检测 + 相似度。

用法: python scripts/analyze_runs.py <截图文件名...> [--top N] [--ascii]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import yaml
from pathlib import Path

from fgo_bot.vision import TemplateMatcher, read_image
from fgo_bot.selfcheck import detect_skill_confirmation_fault


def ascii_map(img, cell=48):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for row in range(0, img.shape[0], cell):
        line = ""
        for col in range(0, img.shape[1], cell):
            blk = hsv[row:row + cell, col:col + cell]
            hc, sc, vc = blk[:, :, 0], blk[:, :, 1], blk[:, :, 2]
            blue = ((hc >= 80) & (hc <= 115) & (sc >= 80) & (vc >= 80)).mean()
            yellow = ((hc >= 15) & (hc <= 35) & (sc >= 120) & (vc >= 120)).mean()
            red = (((hc <= 10) | (hc >= 165)) & (sc >= 80) & (vc >= 80)).mean()
            white = ((sc <= 60) & (vc >= 200)).mean()
            dark = (vc < 90).mean()
            if blue > 0.25: ch = "B"
            elif yellow > 0.25: ch = "Y"
            elif red > 0.25: ch = "R"
            elif white > 0.25: ch = "W"
            elif dark > 0.55: ch = "."
            else: ch = " "
            line += ch
        print(f"  {row:4d} {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--ascii", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "profiles" / "user.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    matcher = TemplateMatcher(cfg, cfg_path)

    from fgo_bot.strategy import BattlePlanner
    planner = BattlePlanner(cfg, cfg_path)

    imgs = {}
    for f in args.files:
        p = Path(f)
        if not p.is_absolute():
            p = root / "runs" / p
        imgs[f] = read_image(p)

    for name, img in imgs.items():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scores = matcher.scores(img)
        dlg = detect_skill_confirmation_fault(img, cfg)
        tgt = planner.skill_target_points(img)
        print(f"== {name}  {img.shape[1]}x{img.shape[0]}  "
              f"mean={gray.mean():.1f} dark={(gray < 30).mean():.1%}")
        for n, s in scores[: args.top]:
            mark = " <<<" if s >= 0.88 else ""
            print(f"    {n:40s} {s:.3f}{mark}")
        print(f"    skill_dialog_heuristic={dlg is not None}  "
              f"target_points={tgt}")
        if args.ascii:
            ascii_map(img)

    if len(imgs) > 1:
        print("\npairwise mse:")
        names = list(imgs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = cv2.cvtColor(imgs[names[i]], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(imgs[names[j]], cv2.COLOR_BGR2GRAY)
                mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
                print(f"  {names[i][:28]:28s} vs {names[j][:28]:28s} mse={mse:9.1f}")


if __name__ == "__main__":
    main()
