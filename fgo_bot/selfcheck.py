"""故障自检与自我纠错模块。

机器人遇到故障时（界面识别失败、卡死、弹窗、断线、游戏退出前台），
不再立刻暂停等待人工，而是先对故障画面做本地模板分析：

1. 黑屏启发式 + 故障模板库分类，识别常见故障画面；
2. 执行映射的修正动作（点弹窗 / 按返回 / 重启游戏 / 重启模拟器）；
3. 每步之后验证：故障模板不再命中，且主规则重新命中已知画面；
4. 逐步升级，全部失败则回退为原有的"暂停等待人工"流程。

所有识别均为本地 OpenCV 模板匹配，不联网、不调用外部服务。
维护公告、战败复活、体力不足等画面应配置为 pause_always，
绝不自动消耗圣晶石、苹果等道具。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .adb import AdbError, MuMuDevice
from .config import load_config
from .vision import MatchResult, TemplateMatcher, annotate_match, write_image

# 故障规则支持的修正动作。
FAULT_ACTIONS = {
    "tap_center",
    "tap_point",
    "key_back",
    "restart_app",
    "restart_emulator",
    "pause_always",
}

# 自检配置默认值。运行时与配置文件的 selfcheck 段合并，
# 旧配置文件缺少该段时也能工作。
SELFCHECK_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_attempts": 3,              # 单次自检修正动作轮数上限
    "max_consecutive_recoveries": 3,  # 连续恢复上限，防死循环
    "restart_app": False,           # 自检排故时不再自动重启游戏（默认关闭）
    "restart_emulator": False,      # 自检排故时不再自动重启模拟器（默认关闭）
    "back_press_first": True,       # 通用阶梯优先按返回键
    "black_screen_max_mean": 12.0,  # 灰度均值低于该值视为黑屏
    "verify_wait_seconds": 2.0,     # 每步验证之间的等待
    "verify_polls": 3,              # 每步验证最多轮询次数
    "action_wait_seconds": 2.0,     # 修正动作后的等待
    "faults": [],                   # 故障规则列表
    # 页面分析回退：未知画面时先对所有主规则打分，置信度高则直接点击。
    # 只允许点击动作简单安全的规则（tap_match / tap_point）；战斗、助战
    # 等复杂动作的模板即使高分也不直接点击，防止在未知画面上乱点。
    "analyze_page_first": True,
    "analyze_fallback_threshold": 0.80,
    "analyze_min_gap": 0.08,
}

# 页面分析回退允许直接点击的简单动作；其余动作（battle / select_support
# 等）在未知画面上点击其模板位置可能产生干扰操作，一律不参与回退。
SAFE_FALLBACK_ACTIONS = {"tap_match", "tap_point"}


@dataclass
class SelfCheckResult:
    recovered: bool
    fault_name: str | None = None
    actions_taken: list[str] = field(default_factory=list)
    restarted_app: bool = False
    restarted_emulator: bool = False
    snapshot: Path | None = None


def load_selfcheck_settings(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """现场重读配置文件，返回 (config, selfcheck设置)。

    每次故障时重读，控制面板里修改的自检开关在下一次故障立即
    生效，无需重启自动推进。读取失败返回 None（安全回退）。
    """
    try:
        config, _ = load_config(config_path)
    except Exception:
        return None
    settings = dict(SELFCHECK_DEFAULTS)
    raw = config.get("selfcheck") or {}
    for key, value in raw.items():
        if key == "faults":
            if isinstance(value, list):
                settings["faults"] = [
                    dict(item)
                    for item in value
                    if isinstance(item, dict) and item.get("name")
                ]
        elif key in settings:
            settings[key] = value
    return config, settings


def build_fault_matcher(
    config: dict[str, Any],
    config_path: Path,
    settings: dict[str, Any],
) -> TemplateMatcher:
    """用 selfcheck.faults 构建独立的故障模板匹配器。"""
    faults = settings.get("faults") or []
    return TemplateMatcher(
        {"screen": config["screen"], "rules": faults},
        config_path,
    )


def classify_fault(
    screen: np.ndarray,
    settings: dict[str, Any],
    fault_matcher: TemplateMatcher,
) -> tuple[str | None, dict[str, Any] | None, MatchResult | None]:
    """识别故障画面，返回 (故障名, 故障规则, 命中)。

    黑屏用灰度均值启发式（伪故障，动作 restart_app），
    其余交给故障模板匹配器。无命中返回三个 None。
    """
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) <= float(settings["black_screen_max_mean"]):
        return (
            "black_screen",
            {"name": "black_screen", "action": "restart_app"},
            None,
        )
    match = fault_matcher.find_first(screen)
    if match is not None:
        return str(match.rule["name"]), match.rule, match
    return None, None, None


def detect_skill_confirmation_fault(
    screen: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """识别战斗中的“技能使用”确认弹窗（无模板，按白色按钮特征判断）。

    弹窗在“决定”点击无效时会一直挡住战斗画面。自检发现该弹窗后
    优先点击“返回”按钮直接关闭，而不是按返回键或重启游戏。
    返回合成故障规则（action=tap_point，point 为基准坐标），
    画面中未识别到弹窗时返回 None。
    """
    battle = config.get("battle") or {}
    base_width = float(config["screen"]["base_width"])
    base_height = float(config["screen"]["base_height"])

    def region_metrics(
        raw_region: list[float],
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in raw_region]
        x1 = max(0, round(x1 * screen.shape[1] / base_width))
        x2 = min(screen.shape[1], round(x2 * screen.shape[1] / base_width))
        y1 = max(0, round(y1 * screen.shape[0] / base_height))
        y2 = min(screen.shape[0], round(y2 * screen.shape[0] / base_height))
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = (
            (hsv[:, :, 1] <= int(
                battle.get("skill_confirm_white_max_saturation", 45)
            ))
            & (hsv[:, :, 2] >= int(
                battle.get("skill_confirm_white_min_value", 170)
            ))
        )
        dark = hsv[:, :, 2] <= int(
            battle.get("skill_confirm_dark_max_value", 80)
        )
        return (
            float(np.count_nonzero(white)) / float(white.size),
            float(np.count_nonzero(dark)) / float(dark.size),
        )

    confirm_white, _ = region_metrics(
        battle.get("skill_confirm_button_region", [825, 480, 1275, 575])
    )
    cancel_white, _ = region_metrics(
        battle.get("skill_cancel_button_region", [325, 480, 735, 575])
    )
    _, panel_dark = region_metrics(
        battle.get("skill_confirm_panel_region", [170, 180, 1430, 600])
    )
    _, title_dark = region_metrics(
        battle.get("skill_confirm_title_region", [600, 185, 1000, 270])
    )
    if (
        confirm_white
        < float(battle.get("skill_confirm_button_min_white_ratio", 0.55))
        or cancel_white
        < float(battle.get("skill_cancel_button_min_white_ratio", 0.55))
        or panel_dark
        < float(battle.get("skill_confirm_panel_min_dark_ratio", 0.60))
        or title_dark
        < float(battle.get("skill_confirm_title_min_dark_ratio", 0.70))
    ):
        return None
    cancel_region = battle.get(
        "skill_cancel_button_region",
        [325, 480, 735, 575],
    )
    return {
        "name": "skill_confirmation_dialog",
        "action": "tap_point",
        "point": [
            (float(cancel_region[0]) + float(cancel_region[2])) / 2.0,
            (float(cancel_region[1]) + float(cancel_region[3])) / 2.0,
        ],
    }


def detect_skill_target_fault(
    screen: np.ndarray,
    config: dict[str, Any],
    main_matcher: TemplateMatcher | None,
    config_path: Path,
) -> dict[str, Any] | None:
    """识别战斗中的技能选人界面（请选择对象），返回点击最右头像的故障规则。

    选人界面模板（skill_target_support）在部分界面上只有约 0.36 分，
    因此用组合特征判断：三张选人头像齐全 + 技能确认弹窗不在 +
    攻击按钮不在（选人界面会遮住攻击按钮）。命中后点击最右侧的
    头像卡（主力输出位），而不是按返回键关掉选人界面。
    """
    if detect_skill_confirmation_fault(screen, config) is not None:
        return None
    from .strategy import BattlePlanner

    planner = BattlePlanner(config, config_path)
    if len(planner.skill_target_points(screen)) < 3:
        return None
    attack_rule = next(
        (
            rule
            for rule in config.get("rules", [])
            if rule.get("name") == "battle_attack"
        ),
        None,
    )
    if (
        main_matcher is not None
        and attack_rule is not None
        and main_matcher.match_rule(screen, attack_rule) is not None
    ):
        return None
    battle = config.get("battle") or {}
    layouts = battle.get(
        "skill_target_layout_points",
        {
            "1": [[800, 470]],
            "2": [[610, 470], [990, 470]],
            "3": [[420, 470], [800, 470], [1180, 470]],
        },
    )
    raw_points = layouts.get("3")
    if not raw_points:
        return None
    rightmost = raw_points[-1]
    return {
        "name": "skill_target_selection",
        "action": "tap_point",
        "point": [float(rightmost[0]), float(rightmost[1])],
    }


def is_pause_always(rule: dict[str, Any] | None) -> bool:
    return rule is not None and str(rule.get("action", "")) == "pause_always"


class SelfChecker:
    def __init__(
        self,
        device: MuMuDevice,
        main_matcher: TemplateMatcher,
        config_path: Path,
        snapshot_dir: Path,
        expected_package: str,
        stop_event: Any,
        dry_run: bool,
        log: Callable[[str], None],
    ) -> None:
        self.device = device
        self.main_matcher = main_matcher
        self.config_path = config_path
        self.snapshot_dir = snapshot_dir
        self.expected_package = str(expected_package or "").strip()
        self.stop_event = stop_event
        self.dry_run = dry_run
        self.log = log
        self._consecutive_recoveries = 0

    @property
    def consecutive_recoveries(self) -> int:
        return self._consecutive_recoveries

    def note_healthy(self) -> None:
        """完成一次真实动作后清零连续恢复计数（状态机确实在前进）。"""
        self._consecutive_recoveries = 0

    # ---- 配置与现场重读 ----

    def _load_settings(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        return load_selfcheck_settings(self.config_path)

    def _build_fault_matcher(
        self,
        config: dict[str, Any],
        settings: dict[str, Any],
    ) -> TemplateMatcher:
        return build_fault_matcher(config, self.config_path, settings)

    # ---- 分类 ----

    def _classify(
        self,
        screen: np.ndarray,
        settings: dict[str, Any],
        fault_matcher: TemplateMatcher,
    ) -> tuple[str | None, dict[str, Any] | None, MatchResult | None]:
        return classify_fault(screen, settings, fault_matcher)

    def _is_pause_always(self, rule: dict[str, Any] | None) -> bool:
        return is_pause_always(rule)

    # ---- 截图与存证 ----

    def _stamp(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

    def _snapshot(
        self,
        screen: np.ndarray,
        label: str,
        match: MatchResult | None = None,
    ) -> Path:
        safe_label = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in label
        )
        path = self.snapshot_dir / f"{self._stamp()}-{safe_label}.png"
        output = annotate_match(screen, match) if match is not None else screen
        write_image(path, output)
        return path

    def _safe_capture(self) -> np.ndarray | None:
        try:
            return self.device.capture()
        except AdbError as exc:
            self.log(f"自检截图失败：{exc}")
            return None

    def _wait(self, seconds: float) -> None:
        self.stop_event.wait(max(0.0, float(seconds)))

    # ---- 修正动作 ----

    def _next_step(
        self,
        fault_rule: dict[str, Any] | None,
        tried: set[str],
        settings: dict[str, Any],
    ) -> tuple[str, str] | None:
        """按顺序挑选下一步修正动作，返回 (日志名, 动作键)；无可用返回 None。

        顺序：故障规则映射的动作 → 按返回键。
        每步只执行一次（tried 去重），逐步升级。
        排故阶段不再自动重启游戏或模拟器，避免破坏性恢复；
        仅当故障规则本身明确配置为 restart_app/restart_emulator 时才执行。
        """
        candidates: list[tuple[str, str]] = []
        if fault_rule is not None:
            action = str(fault_rule.get("action", ""))
            if action in FAULT_ACTIONS and not self._is_pause_always(fault_rule):
                candidates.append((f"故障动作 {action}", action))
        if bool(settings.get("back_press_first", True)):
            candidates.append(("按返回键", "key_back"))
        for label, key in candidates:
            if key not in tried:
                return label, key
        return None

    def _perform(
        self,
        key: str,
        rule: dict[str, Any] | None,
        fault_match: MatchResult | None,
        screen: np.ndarray,
        config: dict[str, Any],
    ) -> None:
        if key == "tap_center":
            if fault_match is None:
                return
            offset = rule.get("offset", [0, 0]) if rule else [0, 0]
            base = config["screen"]
            scale = screen.shape[1] / float(base["base_width"])
            point = (
                fault_match.center[0] + round(float(offset[0]) * scale),
                fault_match.center[1] + round(float(offset[1]) * scale),
            )
            self.device.tap(*point)
        elif key == "tap_point":
            point = (rule or {}).get("point") or [800, 450]
            base = config["screen"]
            point = (
                round(float(point[0]) * screen.shape[1] / float(base["base_width"])),
                round(float(point[1]) * screen.shape[0] / float(base["base_height"])),
            )
            self.device.tap(*point)
        elif key == "key_back":
            self.device.keyevent(4)
        elif key == "restart_app":
            self.device.restart_app(self.expected_package)
        elif key == "restart_emulator":
            self.device.restart_emulator(should_stop=self.stop_event.is_set)
            if self.expected_package:
                self.device.start_app(self.expected_package)

    # ---- 页面分析回退（自检第一步） ----

    def _analyze_page_fallback(
        self,
        screen: np.ndarray,
        config: dict[str, Any],
        settings: dict[str, Any],
    ) -> SelfCheckResult | None:
        """未知画面时对所有主规则回溯打分，置信度高则直接点击对应位置。

        当主匹配器没有命中任何规则时，可能是因为该帧质量略差或阈值
        偏高导致漏检。此时对所有规则放宽阈值重新打分，若某个规则的
        得分显著高于其他规则且超过回退阈值，则极可能是该规则对应的
        画面，直接点击其中心位置尝试恢复推进。
        """
        if not bool(settings.get("analyze_page_first", True)):
            return None

        all_scores = self.main_matcher.scores(screen)
        valid_names = {
            r["name"]
            for r in config.get("rules", [])
            if r.get("enabled", True)
            and not r.get("detect_only", False)
            and str(r.get("action", "")) in SAFE_FALLBACK_ACTIONS
        }
        scores = [(name, s) for name, s in all_scores if name in valid_names]
        if not scores:
            return None

        threshold = float(settings.get("analyze_fallback_threshold", 0.45))
        min_gap = float(settings.get("analyze_min_gap", 0.05))

        best_name, best_score = scores[0]
        if best_score < threshold:
            self.log(
                f"自检页面分析：最高分 {best_name}={best_score:.3f} 低于阈值 "
                f"{threshold}，不执行回退点击"
            )
            return None

        if len(scores) > 1:
            second_score = scores[1][1]
            if best_score - second_score < min_gap:
                self.log(
                    f"自检页面分析：第一名 {best_name}={best_score:.3f} 与第二名 "
                    f"{scores[1][0]}={second_score:.3f} 差距不足 {min_gap}，"
                    f"置信度不够，不执行回退点击"
                )
                return None

        rule = next(
            (r for r in config.get("rules", []) if r.get("name") == best_name),
            None,
        )
        if rule is None:
            return None

        # 先用正常阈值获取匹配位置；若失败再用放宽阈值兜底
        match = self.main_matcher.match_rule(screen, rule)
        if match is None:
            relaxed = dict(rule)
            relaxed["threshold"] = -1
            match = self.main_matcher.match_rule(screen, relaxed)
        if match is None:
            return None

        self.log(
            f"自检页面分析：识别为 {best_name}（{best_score:.3f}），"
            f"尝试直接点击对应位置恢复"
        )

        offset = rule.get("offset", [0, 0]) if rule else [0, 0]
        base = config["screen"]
        scale = screen.shape[1] / float(base["base_width"])
        point = (
            match.center[0] + round(float(offset[0]) * scale),
            match.center[1] + round(float(offset[1]) * scale),
        )
        try:
            self.device.tap(*point)
        except AdbError as exc:
            self.log(f"自检页面分析点击失败：{exc}")
            return None

        # 验证：等待后截图，看主规则是否重新命中已知画面。
        # 先用正常阈值 find_first 验证；若未命中（比如某些规则 threshold
        # 偏高导致漏检），再用 scores 回退验证。回退验证若发现高置信度
        # 的可点击规则（如 skip_confirm），会直接连锁点击，避免主循环
        # 因阈值问题再次漏检而反复进入未知画面。
        action_wait = float(settings.get("action_wait_seconds", 2.0))
        verify_wait = float(settings.get("verify_wait_seconds", 2.0))
        verify_polls = max(1, int(settings.get("verify_polls", 3)))
        self._wait(action_wait)
        # 已点击过的规则集合，防止重复点击同一位置
        verify_exclude: set[str] = {best_name}

        for _ in range(verify_polls):
            if self.stop_event.is_set():
                return None
            current = self._safe_capture()
            if current is None:
                return None
            main_match = self.main_matcher.find_first(
                current, exclude=verify_exclude
            )
            if main_match is not None:
                self._consecutive_recoveries += 1
                self.log(
                    f"自检页面分析成功：点击 {best_name} 后画面切换到 "
                    f"{main_match.rule['name']}；连续恢复 "
                    f"{self._consecutive_recoveries} 次"
                )
                return SelfCheckResult(
                    recovered=True,
                    fault_name=f"analyze_fallback:{best_name}",
                    actions_taken=[f"页面分析回退点击 {best_name}"],
                )
            # 回退验证：scores 检查其他规则是否置信度高
            fallback_scores = self.main_matcher.scores(current)
            fallback_valid = [
                (n, s)
                for n, s in fallback_scores
                if n not in verify_exclude and n in valid_names
            ]
            if fallback_valid:
                fb_name, fb_score = fallback_valid[0]
                if fb_score >= threshold:
                    fb_gap_ok = (
                        len(fallback_valid) == 1
                        or fallback_valid[1][1] <= fb_score - min_gap
                    )
                    if fb_gap_ok:
                        fb_rule = next(
                            (
                                r
                                for r in config.get("rules", [])
                                if r.get("name") == fb_name
                            ),
                            None,
                        )
                        # 若该规则是可点击的简单动作，直接连锁点击，
                        # 而不是把难题扔回给主循环（防止主循环 threshold
                        # 偏高再次漏检）。
                        if fb_rule is not None and fb_rule.get(
                            "action"
                        ) in {
                            "tap_match",
                            "tap_point",
                        }:
                            fb_match = self.main_matcher.match_rule(
                                current, fb_rule
                            )
                            if fb_match is None:
                                relaxed = dict(fb_rule)
                                relaxed["threshold"] = -1
                                fb_match = self.main_matcher.match_rule(
                                    current, relaxed
                                )
                            if fb_match is not None:
                                offset = fb_rule.get("offset", [0, 0])
                                base = config["screen"]
                                scale = (
                                    current.shape[1]
                                    / float(base["base_width"])
                                )
                                point = (
                                    fb_match.center[0]
                                    + round(float(offset[0]) * scale),
                                    fb_match.center[1]
                                    + round(float(offset[1]) * scale),
                                )
                                try:
                                    self.device.tap(*point)
                                except AdbError:
                                    pass
                                else:
                                    self.log(
                                        f"自检页面分析连锁点击："
                                        f"{fb_name}（{fb_score:.3f}）"
                                    )
                                    verify_exclude.add(fb_name)
                                    self._wait(action_wait)
                                    continue
                        # 不可点击或点击失败时，直接宣布恢复成功，
                        # 让主循环尝试处理。
                        self._consecutive_recoveries += 1
                        self.log(
                            f"自检页面分析成功（回退验证）："
                            f"点击 {best_name} 后画面疑似 "
                            f"{fb_name}（{fb_score:.3f}）；"
                            f"连续恢复 {self._consecutive_recoveries} 次"
                        )
                        return SelfCheckResult(
                            recovered=True,
                            fault_name=f"analyze_fallback:{best_name}",
                            actions_taken=[
                                f"页面分析回退点击 {best_name}"
                            ],
                        )
            self._wait(verify_wait)

        self.log(
            f"自检页面分析：点击 {best_name} 后未验证通过，继续常规自检"
        )
        return None

    # ---- 主入口：故障自检 ----

    def try_recover(
        self,
        reason: str,
        screen: np.ndarray,
        match: MatchResult | None = None,
        *,
        in_battle: bool = True,
    ) -> SelfCheckResult | None:
        """对故障画面执行自检与自我纠错。

        in_battle 为调用方的战斗流程状态：技能确认/选人等战斗启发式
        只在战斗画面生效，剧情演出画面上的三张立绘不会再被误判成
        技能选人界面而乱点。返回 SelfCheckResult 表示已恢复（调用方抛
        RecoveredAction 继续主循环）；返回 None 表示无法自动恢复，
        调用方照旧走暂停流程。
        """
        if self.dry_run or self.stop_event.is_set():
            return None
        loaded = self._load_settings()
        if loaded is None:
            self.log("自检配置读取失败，回退为暂停等待人工处理")
            return None
        config, settings = loaded
        if not bool(settings.get("enabled", True)):
            return None
        max_streak = max(1, int(settings["max_consecutive_recoveries"]))
        if self._consecutive_recoveries >= max_streak:
            self.log(
                f"连续自动恢复已达 {self._consecutive_recoveries} 次上限，"
                "本次改为暂停等待人工处理"
            )
            return None

        # 自检第一步：未知画面时先全面分析页面，回溯所有主规则打分，
        # 若置信度高则直接点击对应位置尝试恢复。
        analyze_result = self._analyze_page_fallback(screen, config, settings)
        if analyze_result is not None and analyze_result.recovered:
            return analyze_result

        fault_matcher = self._build_fault_matcher(config, settings)
        fault_name, rule, fault_match = self._classify(
            screen,
            settings,
            fault_matcher,
        )
        if fault_name is None and in_battle:
            # 模板库之外的内置启发式：先看画面是不是战斗中的技能
            # 确认弹窗。是的话直接点”返回”关闭，避免重启游戏。
            # 只在战斗流程中检测：剧情演出画面上的立绘会被误判成
            # 选人界面，非战斗画面不执行这两条启发式。
            dialog_rule = detect_skill_confirmation_fault(screen, config)
            if dialog_rule is not None:
                fault_name = str(dialog_rule["name"])
                rule = dialog_rule
                fault_match = None
        if fault_name is None and in_battle:
            # 再看是不是技能选人界面：是的话点最右侧头像卡
            # 选择主力输出位，而不是按返回键取消选人。
            target_rule = detect_skill_target_fault(
                screen,
                config,
                self.main_matcher,
                self.config_path,
            )
            if target_rule is not None:
                fault_name = str(target_rule["name"])
                rule = target_rule
                fault_match = None
        result = SelfCheckResult(
            recovered=False,
            fault_name=fault_name,
            snapshot=self._snapshot(
                screen,
                f"FAULT-{fault_name or 'unknown'}",
                fault_match,
            ),
        )
        self.log(
            f"自检开始：{reason}；识别故障：{fault_name or '未知画面'}；"
            f"现场截图：{result.snapshot}"
        )
        if self._is_pause_always(rule):
            self.log(
                f"故障 {fault_name} 配置为 pause_always，不自动处理，等待人工"
            )
            return None

        max_attempts = max(1, int(settings["max_attempts"]))
        tried: set[str] = set()
        current = screen
        for attempt in range(max_attempts):
            step = self._next_step(rule, tried, settings)
            if step is None:
                break
            label, key = step
            self.log(f"自检执行：{label}（第 {attempt + 1}/{max_attempts} 步）")
            tried.add(key)
            try:
                self._perform(key, rule, fault_match, current, config)
            except AdbError as exc:
                self.log(f"自检动作 {label} 失败：{exc}")
                result.actions_taken.append(f"{label}失败")
                current = self._safe_capture()
                if current is None:
                    return None
                continue
            result.actions_taken.append(label)
            if key == "restart_app":
                result.restarted_app = True
            elif key == "restart_emulator":
                result.restarted_emulator = True
            self._wait(float(settings["action_wait_seconds"]))
            if self.stop_event.is_set():
                return None

            # 验证：故障消失且主规则命中 → 恢复；故障仍在 → 升级下一步。
            for _ in range(max(1, int(settings.get("verify_polls", 3)))):
                if self.stop_event.is_set():
                    return None
                current = self._safe_capture()
                if current is None:
                    return None
                new_name, new_rule, new_match = self._classify(
                    current,
                    settings,
                    fault_matcher,
                )
                if new_name is None and in_battle:
                    # 弹窗仍在时视为故障未解除，不能因 battle_attack
                    # 模板穿过弹窗误匹配就宣布恢复。
                    dialog_rule = detect_skill_confirmation_fault(
                        current,
                        config,
                    )
                    if dialog_rule is not None:
                        new_name = str(dialog_rule["name"])
                        new_rule = dialog_rule
                        new_match = None
                if new_name is None and in_battle:
                    # 选人界面未消失时同样不能宣布恢复（攻击按钮
                    # 被遮住本就不会误匹配，此处主要防止漏判）。
                    target_rule = detect_skill_target_fault(
                        current,
                        config,
                        self.main_matcher,
                        self.config_path,
                    )
                    if target_rule is not None:
                        new_name = str(target_rule["name"])
                        new_rule = target_rule
                        new_match = None
                if new_name is not None:
                    fault_name, rule, fault_match = new_name, new_rule, new_match
                    if self._is_pause_always(rule):
                        self.log(
                            f"识别到 {fault_name}，配置为 pause_always，"
                            "停止自动恢复"
                        )
                        return None
                    break
                # 验证时排除"刚失败的那条规则"：否则点击无效时同一条
                # 规则再次命中会被误判为"已恢复"，形成"恢复→失败→恢复"
                # 的循环（如战斗详情关闭按钮连点无效的场景）。
                verify_exclude = (
                    {match.rule["name"]} if match is not None else None
                )
                main_match = self.main_matcher.find_first(
                    current,
                    exclude=verify_exclude,
                )
                if main_match is not None:
                    result.recovered = True
                    self._consecutive_recoveries += 1
                    self.log(
                        f"自检成功：{reason}；已恢复，"
                        f"当前画面 {main_match.rule['name']}；"
                        f"连续恢复 {self._consecutive_recoveries} 次"
                    )
                    return result
                self._wait(float(settings["verify_wait_seconds"]))

        self.log(
            f"自检未恢复：{reason}；已执行：{'、'.join(result.actions_taken) or '无'}"
        )
        return None

    # ---- ADB 故障恢复（run() 的 AdbError 分支） ----

    def recover_adb(self, reason: str) -> SelfCheckResult | None:
        """ADB 层面故障：重连并重试截图，必要时重启模拟器。"""
        if self.dry_run or self.stop_event.is_set():
            return None
        loaded = self._load_settings()
        if loaded is None or not bool(loaded[1].get("enabled", True)):
            return None
        settings = loaded[1]
        max_streak = max(1, int(settings["max_consecutive_recoveries"]))
        if self._consecutive_recoveries >= max_streak:
            return None

        self.log(f"自检开始：{reason}；先尝试重连 ADB")
        try:
            self.device.connect(force=True)
            frame = self.device.capture()
        except AdbError:
            frame = None
        if frame is not None:
            self._consecutive_recoveries += 1
            self.log(f"自检成功：{reason}；ADB 重连成功，继续运行")
            return SelfCheckResult(
                recovered=True,
                actions_taken=["ADB 重连"],
            )
        if not bool(settings.get("restart_emulator", False)):
            return None
        self.log("ADB 重连失败，尝试重启模拟器")
        try:
            self.device.restart_emulator(should_stop=self.stop_event.is_set)
            if self.expected_package:
                self.device.start_app(self.expected_package)
            self.device.capture()
        except AdbError as exc:
            self.log(f"ADB 自检失败：{exc}")
            return None
        self._consecutive_recoveries += 1
        return SelfCheckResult(
            recovered=True,
            actions_taken=["重启模拟器并重新连接"],
            restarted_emulator=True,
        )
