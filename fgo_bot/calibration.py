from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .adb import MuMuDevice
from .config import get_rule, resolve_from_config, save_config
from .vision import read_image, write_image


class CalibrationCancelled(RuntimeError):
    pass


def _select_template(screen, title: str):
    x, y, width, height = cv2.selectROI(
        title,
        screen,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyAllWindows()
    if width <= 1 or height <= 1:
        raise CalibrationCancelled("未选择有效区域")
    return screen[y : y + height, x : x + width]


def calibrate_rule(
    device: MuMuDevice,
    config: dict[str, Any],
    config_path: Path,
    rule_name: str,
    *,
    input_path: Path | None = None,
) -> Path:
    rule = get_rule(config, rule_name)
    screen = read_image(input_path) if input_path else device.capture()

    title = f"框选 {rule_name} 的稳定特征，Enter/Space确认，C取消"
    crop = _select_template(screen, title)
    target = resolve_from_config(config_path, rule["template"])
    write_image(target, crop)
    rule["enabled"] = True
    save_config(config, config_path)
    return target


def calibrate_enemy_class(
    device: MuMuDevice,
    config: dict[str, Any],
    config_path: Path,
    class_name: str,
    *,
    input_path: Path | None = None,
) -> Path:
    screen = read_image(input_path) if input_path else device.capture()
    crop = _select_template(
        screen,
        f"框选敌方 {class_name} 职阶图标，Enter/Space确认，C取消",
    )
    relative = Path("templates") / "enemy_classes" / f"{class_name}.png"
    target = (config_path.parent / relative).resolve()
    write_image(target, crop)
    config.setdefault("support", {}).setdefault(
        "enemy_class_templates", {}
    )[class_name] = relative.as_posix()
    save_config(config, config_path)
    return target


def calibrate_battle_enemy_class(
    device: MuMuDevice,
    config: dict[str, Any],
    config_path: Path,
    class_name: str,
    *,
    input_path: Path | None = None,
) -> Path:
    screen = read_image(input_path) if input_path else device.capture()
    crop = _select_template(
        screen,
        f"框选战斗中敌方 {class_name} 职阶图标，Enter/Space确认，C取消",
    )
    relative = Path("templates") / "battle_classes" / f"{class_name}.png"
    target = (config_path.parent / relative).resolve()
    write_image(target, crop)
    config.setdefault("battle", {}).setdefault(
        "enemy_class_templates", {}
    )[class_name] = relative.as_posix()
    save_config(config, config_path)
    return target


def calibrate_support_candidate(
    device: MuMuDevice,
    config: dict[str, Any],
    config_path: Path,
    *,
    candidate_id: str,
    name: str,
    class_name: str,
    np_color: str,
    servant_level: int,
    priority: int,
    party_slot: int,
    input_path: Path | None = None,
) -> Path:
    if not candidate_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in candidate_id
    ):
        raise ValueError("助战 ID 只能包含英文字母、数字、下划线和连字符")
    screen = read_image(input_path) if input_path else device.capture()
    crop = _select_template(
        screen,
        f"框选助战 {name} 的好友名/NP5标记及从者特征，Enter/Space确认，C取消",
    )
    relative = Path("templates") / "supports" / f"{candidate_id}.png"
    target = (config_path.parent / relative).resolve()
    write_image(target, crop)

    support = config.setdefault("support", {})
    candidates = support.setdefault("candidates", [])
    existing = next(
        (
            candidate
            for candidate in candidates
            if str(candidate.get("id")) == candidate_id
        ),
        None,
    )
    candidate = existing if existing is not None else {}
    candidate.update(
        {
            "id": candidate_id,
            "name": name,
            "class": class_name,
            "level": int(servant_level),
            "np_level": 5,
            "np_target": "aoe",
            "np_color": np_color,
            "party_slot": int(party_slot),
            "priority": int(priority),
            "enabled": True,
            "template": relative.as_posix(),
        }
    )
    if existing is None:
        candidate.setdefault("opening_steps", [])
        candidates.append(candidate)
    save_config(config, config_path)
    return target


FAULT_ACTIONS = {
    "tap_center",
    "tap_point",
    "key_back",
    "restart_app",
    "restart_emulator",
    "pause_always",
}


def calibrate_fault(
    device: MuMuDevice,
    config: dict[str, Any],
    config_path: Path,
    fault_name: str,
    action: str,
    *,
    point: list[int] | None = None,
    threshold: float | None = None,
    input_path: Path | None = None,
) -> Path:
    """框选一个故障画面特征，保存模板并写入 selfcheck.faults。"""
    if not fault_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in fault_name
    ):
        raise ValueError("故障名只能包含英文字母、数字、下划线和连字符")
    if action not in FAULT_ACTIONS:
        raise ValueError(
            f"动作必须是以下之一：{'、'.join(sorted(FAULT_ACTIONS))}"
        )
    if action == "tap_point" and point is None:
        raise ValueError("动作 tap_point 需要 --point x,y 指定点击坐标")
    screen = read_image(input_path) if input_path else device.capture()
    crop = _select_template(
        screen,
        f"框选故障画面 {fault_name} 的稳定特征，Enter/Space确认，C取消",
    )
    relative = Path("templates") / "faults" / f"{fault_name}.png"
    target = (config_path.parent / relative).resolve()
    write_image(target, crop)

    faults = config.setdefault("selfcheck", {}).setdefault("faults", [])
    existing = next(
        (
            fault
            for fault in faults
            if str(fault.get("name")) == fault_name
        ),
        None,
    )
    fault = existing if existing is not None else {}
    fault.update(
        {
            "name": fault_name,
            "template": relative.as_posix(),
            "action": action,
        }
    )
    if point is not None:
        fault["point"] = [int(point[0]), int(point[1])]
    if threshold is not None:
        fault["threshold"] = float(threshold)
    if existing is None:
        faults.append(fault)
    save_config(config, config_path)
    return target
