from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .adb import AdbError, FramePrefetcher, MuMuDevice
from .config import resolve_from_config
from .selfcheck import SelfCheckResult, SelfChecker
from .strategy import (
    BattlePlanner,
    BattleVisionError,
    SupportChoice,
    SupportSelector,
    VisualMatch,
)
from .vision import MatchResult, TemplateMatcher, annotate_match, write_image


class PauseRequested(RuntimeError):
    def __init__(self, reason: str, snapshot: Path | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.snapshot = snapshot


class RecoveredAction(RuntimeError):
    """自检已自动恢复故障；只在 _pause 抛出、只在 run() 捕获。

    中断当前半执行的动作，回到主循环重新识别画面后继续推进。
    """

    def __init__(self, reason: str, result: SelfCheckResult) -> None:
        super().__init__(reason)
        self.reason = reason
        self.result = result


LogCallback = Callable[[str], None]


class BotRunner:
    def __init__(
        self,
        device: MuMuDevice,
        matcher: TemplateMatcher,
        config: dict[str, Any],
        config_path: Path,
        *,
        dry_run: bool,
        log: LogCallback = print,
        on_recovered: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.matcher = matcher
        self.config = config
        self.config_path = config_path
        self.dry_run = dry_run
        self.log = log
        self.on_recovered = on_recovered
        self.stop_event = threading.Event()
        self.support_selector = SupportSelector(config, config_path)
        self.battle_planner = BattlePlanner(config, config_path)
        self.enemy_class: str | None = None
        self.selected_support: SupportChoice | None = None
        self.support_refreshes = 0
        self.support_scrolls = 0
        self.support_scroll_stalls = 0
        self.support_filter_class: str | None = None
        self.support_initial_page_scanned = False
        self.support_class_queue: list[str] = []
        self.support_search_levels: list[tuple[int, int]] = []
        self.support_search_rule_keys: list[str] = []
        self.support_class_index = 0
        self.support_recommended_classes: list[str] = []
        self.support_search_phase = "strict"
        self.support_candidate_pass = ""
        self.support_default_filter_class: str | None = None
        self.support_default_guest_seen = False
        self.support_default_normal_seen = False
        self.support_default_scan_complete = False
        self.support_berserker_guest_seen = False
        self.support_berserker_normal_seen = False
        self.support_berserker_scan_complete = False
        self.actions = 0
        self.battle_turn = 0
        self._single_survivor_skills_used = False
        self._aoe_np_fired = False
        self._post_np_charge_used = False
        self._used_skill_groups: set[str] = set()
        self._blocked_np_slots: set[int] = set()
        self._pending_np_slots: set[int] = set()
        self._last_damage_slot: int | None = None
        self._party_state_initialized = False
        self._party_active_slots: set[int] = set()
        self._party_identity_features: dict[int, np.ndarray] = {}
        self._party_generations: dict[int, int] = {}
        self._party_inactive_slots: set[int] = set()
        self._stable_name: str | None = None
        self._stable_count = 0
        self._unknown_since = time.monotonic()
        self._has_seen_recognized_rule = False
        self._unknown_wait_context: str | None = None
        self._battle_flow_active = False
        self._last_foreground_check = 0.0
        self._blocked_rule: str | None = None
        self._blocked_since = 0.0
        self._blocked_fingerprint: np.ndarray | None = None
        self._blocked_center: tuple[int, int] | None = None
        self._last_static_none_fingerprint: np.ndarray | None = None
        self._unknown_static_since = 0.0
        self._unknown_static_fingerprint: np.ndarray | None = None
        self._static_skip_count = 0

        behavior = config["behavior"]
        # 运行模式：story = 主线推进（原行为）；farm = 日常刷自由本。
        self.mode = str(behavior.get("mode", "story")).strip().lower()
        self.farm_mode = self.mode == "farm"
        if self.farm_mode:
            self._exclude_rules: set[str] = {
                "home_main_next",
                "map_next",
                "map_next_arrow_fallback",
                "map_next_gray_fallback",
                "map_next_arrow_soft_center",
                "story_skip",
                "story_skip_light_background",
                "story_skip_gray_text_fallback",
                "skip_confirm",
            }
        else:
            self._exclude_rules = {"free_quest_node"}
        # 战斗专属规则：仅在战斗流程激活（battle_attack / command_cards
        # 等规则命中后）才参与匹配。battle_detail_close 的白色圆形按钮
        # 模板曾在剧情演出画面把立绘的白发误认成按钮并连点 3 次。
        self._battle_only_rules = {"battle_detail_close"}
        # 规则名快速索引（剧情规则优先复检等场景使用）。
        self._rules_by_name = {
            str(rule.get("name")): rule for rule in config["rules"]
        }
        # 刷本状态：节点是否已被点击（决定详情条合成规则点"开始"还是关闭）；
        # farm_round 累计已开始的刷本轮数；抑制窗口防止重复点开始按钮。
        # _farm_confirm_pending：已点详情条"开始任务"、等待出击确认弹窗期间为真；
        # 由其他规则命中（助战/编队/战斗）时清除。确认弹窗模板在部分关卡
        # 只有约 0.74 分，需用组合特征 + 固定坐标兜底，不能依赖普通模板。
        self._farm_node_tapped = False
        self.farm_round = 0
        self._farm_strip_suppress_until = 0.0
        self._farm_confirm_pending = False
        # 刷本次数限制（仅在 farm 模式生效；0 表示无限制）。
        self.farm_max_rounds = 0
        self.farm_limit_reached = False
        if self.farm_mode:
            if bool(behavior.get("farm_limit_enabled", False)):
                self.farm_max_rounds = int(
                    behavior.get("farm_max_rounds", 0)
                )
        # 上一帧执行过的规则名：新帧优先复检它（连续帧相关性高，提速明显）。
        self._last_match_prefer: str | None = None
        self.poll_seconds = float(behavior.get("poll_seconds", 1.0))
        self.stable_frames = int(behavior.get("stable_frames", 2))
        self.unknown_pause_seconds = float(
            behavior.get("unknown_pause_seconds", 25)
        )
        self.startup_unknown_pause_seconds = float(
            behavior.get("startup_unknown_pause_seconds", 120)
        )
        self.battle_animation_pause_seconds = float(
            config.get("battle", {}).get(
                "animation_pause_seconds",
                180,
            )
        )
        self._current_unknown_limit = self.unknown_pause_seconds
        self.max_actions = int(behavior.get("max_actions", 500))
        self.expected_package = str(config["device"].get("package", "")).strip()
        self.snapshot_dir = resolve_from_config(
            config_path,
            behavior.get("snapshot_dir", "../runs"),
        )
        self._prefetcher = FramePrefetcher(device)
        self.selfcheck = SelfChecker(
            device,
            matcher,
            config_path,
            self.snapshot_dir,
            self.expected_package,
            self.stop_event,
            dry_run,
            log,
        )

    def stop(self) -> None:
        self._prefetcher.stop()
        self.stop_event.set()

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
        output = annotate_match(screen, match) if match else screen
        write_image(path, output)
        return path

    def _scaled_point(
        self,
        point: list[float] | tuple[float, float],
        screen: np.ndarray,
    ) -> tuple[int, int]:
        height, width = screen.shape[:2]
        base = self.config["screen"]
        return (
            round(float(point[0]) * width / float(base["base_width"])),
            round(float(point[1]) * height / float(base["base_height"])),
        )

    def _free_quest_match(
        self,
        screen: np.ndarray,
    ) -> MatchResult | None:
        """Recognize the green free-quest detail strip shown over a map."""
        behavior = self.config.get("behavior", {})
        # 刷本模式必须识别详情条（点"开始"或自愈关闭），不受关闭开关影响。
        if not self.farm_mode and not bool(
            behavior.get("auto_close_free_quest", True)
        ):
            return None

        raw_roi = behavior.get(
            "free_quest_header_roi",
            [770, 135, 1545, 230],
        )
        if not isinstance(raw_roi, list) or len(raw_roi) != 4:
            return None
        x1, y1 = self._scaled_point((raw_roi[0], raw_roi[1]), screen)
        x2, y2 = self._scaled_point((raw_roi[2], raw_roi[3]), screen)
        height, width = screen.shape[:2]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        hsv = cv2.cvtColor(screen[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        lower = np.asarray(
            behavior.get("free_quest_header_hsv_lower", [35, 60, 45]),
            dtype=np.uint8,
        )
        upper = np.asarray(
            behavior.get("free_quest_header_hsv_upper", [90, 255, 255]),
            dtype=np.uint8,
        )
        if lower.shape != (3,) or upper.shape != (3,):
            return None
        green_mask = cv2.inRange(hsv, lower, upper)
        green_ratio = float(np.count_nonzero(green_mask)) / float(
            green_mask.size
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            green_mask,
            8,
        )
        component_area_ratio = 0.0
        component_width_ratio = 0.0
        if component_count > 1:
            largest_index = 1 + int(
                np.argmax(stats[1:, cv2.CC_STAT_AREA])
            )
            component_area_ratio = float(
                stats[largest_index, cv2.CC_STAT_AREA]
            ) / float(green_mask.size)
            component_width_ratio = float(
                stats[largest_index, cv2.CC_STAT_WIDTH]
            ) / float(green_mask.shape[1])
        dense_row_ratio = float(
            np.mean(
                np.mean(green_mask > 0, axis=1)
                >= float(
                    behavior.get(
                        "free_quest_header_dense_row_fill_ratio",
                        0.75,
                    )
                )
            )
        )
        header_present = (
            green_ratio
            >= float(
                behavior.get("free_quest_header_min_green_ratio", 0.55)
            )
            and component_area_ratio
            >= float(
                behavior.get(
                    "free_quest_header_min_component_area_ratio",
                    0.55,
                )
            )
            and component_width_ratio
            >= float(
                behavior.get(
                    "free_quest_header_min_component_width_ratio",
                    0.90,
                )
            )
            and dense_row_ratio
            >= float(
                behavior.get(
                    "free_quest_header_min_dense_row_ratio",
                    0.40,
                )
            )
        )
        if not header_present:
            return None

        def light_and_blue_ratios(
            roi_value: list[float] | tuple[float, float, float, float],
        ) -> tuple[float, float]:
            roi_x1, roi_y1 = self._scaled_point(
                (roi_value[0], roi_value[1]),
                screen,
            )
            roi_x2, roi_y2 = self._scaled_point(
                (roi_value[2], roi_value[3]),
                screen,
            )
            roi_x1 = max(0, min(width, roi_x1))
            roi_x2 = max(0, min(width, roi_x2))
            roi_y1 = max(0, min(height, roi_y1))
            roi_y2 = max(0, min(height, roi_y2))
            roi = screen[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                return 0.0, 0.0
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            light = (
                (roi_hsv[:, :, 1] <= 100)
                & (roi_hsv[:, :, 2] >= 145)
            )
            blue = (
                (roi_hsv[:, :, 0] >= 90)
                & (roi_hsv[:, :, 0] <= 140)
                & (roi_hsv[:, :, 1] >= 80)
                & (roi_hsv[:, :, 2] >= 45)
            )
            pixel_count = float(roi_hsv.shape[0] * roi_hsv.shape[1])
            return (
                float(np.count_nonzero(light)) / pixel_count,
                float(np.count_nonzero(blue)) / pixel_count,
            )

        body_roi = behavior.get(
            "free_quest_body_roi",
            [950, 215, 1515, 310],
        )
        close_roi = behavior.get(
            "free_quest_close_roi",
            [0, 0, 220, 115],
        )
        if (
            not isinstance(body_roi, list)
            or len(body_roi) != 4
            or not isinstance(close_roi, list)
            or len(close_roi) != 4
        ):
            return None
        body_light_ratio, _ = light_and_blue_ratios(body_roi)
        close_light_ratio, close_blue_ratio = light_and_blue_ratios(close_roi)
        if (
            body_light_ratio
            < float(behavior.get("free_quest_body_min_light_ratio", 0.30))
            or close_light_ratio
            < float(behavior.get("free_quest_close_min_light_ratio", 0.40))
            or close_blue_ratio
            < float(behavior.get("free_quest_close_min_blue_ratio", 0.015))
        ):
            return None

        if (
            self.farm_mode
            and self._farm_node_tapped
            and time.monotonic() >= self._farm_strip_suppress_until
        ):
            # 刷本模式：节点已点击、详情条在场 → 点"开始任务"进入战斗。
            # 点击点用配置坐标（free_quest_start_point），无需模板。
            start_point = self._scaled_point(
                behavior.get("free_quest_start_point", [1435, 280]),
                screen,
            )
            return MatchResult(
                rule={
                    "name": "free_quest_start",
                    "action": "quest_start",
                    "cooldown": 2,
                    "stable_frames": 2,
                    "unknown_pause_seconds": 60,
                },
                score=min(1.0, green_ratio),
                x=start_point[0] - 1,
                y=start_point[1] - 1,
                width=2,
                height=2,
            )
        close_point = self._scaled_point(
            behavior.get("free_quest_close_point", [115, 55]),
            screen,
        )
        return MatchResult(
            rule={
                "name": "free_quest_close",
                "action": "close_free_quest",
                "cooldown": 0.2,
                "stable_frames": 1,
                "unknown_pause_seconds": 60,
                "reset_quest": True,
                "retry_attempts": int(
                    behavior.get("free_quest_close_retry_attempts", 3)
                ),
                "retry_wait_seconds": float(
                    behavior.get("free_quest_close_retry_wait_seconds", 0.8)
                ),
            },
            score=min(1.0, green_ratio),
            x=close_point[0] - 1,
            y=close_point[1] - 1,
            width=2,
            height=2,
        )

    def _farm_confirm_match(
        self,
        screen: np.ndarray,
    ) -> MatchResult | None:
        """刷本模式识别"出击确认"弹窗，返回点击"开始任务"的合成规则。

        quest_confirm_start 模板在该弹窗上只有约 0.74 分（阈值 0.70 的
        余量太小），故用组合特征兜底：弹窗中部蓝色面板 + 底部左"取消"、
        右"开始任务"两个白色按钮。仅在点击详情条"开始任务"之后
        （_farm_confirm_pending）生效，避免在其他画面误触。
        """
        if not self.farm_mode or not self._farm_confirm_pending:
            return None
        behavior = self.config.get("behavior", {})

        def ratio(
            raw_roi: list[float] | tuple[float, float, float, float],
            kind: str,
        ) -> float:
            x1, y1 = self._scaled_point((raw_roi[0], raw_roi[1]), screen)
            x2, y2 = self._scaled_point((raw_roi[2], raw_roi[3]), screen)
            height, width = screen.shape[:2]
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                return 0.0
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            if kind == "white":
                mask = (hsv[:, :, 1] <= 100) & (hsv[:, :, 2] >= 170)
            else:  # 弹窗蓝色面板
                mask = (
                    (hsv[:, :, 0] >= 90)
                    & (hsv[:, :, 0] <= 130)
                    & (hsv[:, :, 1] >= 60)
                    & (hsv[:, :, 2] >= 60)
                )
            return float(mask.mean())

        right_white = ratio(
            behavior.get("farm_confirm_button_right_roi", [900, 675, 1200, 750]),
            "white",
        )
        left_white = ratio(
            behavior.get("farm_confirm_button_left_roi", [390, 675, 690, 750]),
            "white",
        )
        panel_blue = ratio(
            behavior.get("farm_confirm_panel_roi", [390, 400, 1212, 650]),
            "blue",
        )
        if not (
            right_white
            >= float(behavior.get("farm_confirm_min_right_white", 0.30))
            and left_white
            >= float(behavior.get("farm_confirm_min_left_white", 0.20))
            and panel_blue
            >= float(behavior.get("farm_confirm_min_panel_blue", 0.15))
        ):
            return None
        point = self._scaled_point(
            behavior.get("farm_confirm_point", [1050, 712]),
            screen,
        )
        return MatchResult(
            rule={
                "name": "farm_confirm",
                "action": "farm_confirm",
                "cooldown": 3,
                "stable_frames": 2,
                "unknown_pause_seconds": 60,
            },
            score=min(1.0, max(right_white, left_white, panel_blue)),
            x=point[0] - 1,
            y=point[1] - 1,
            width=2,
            height=2,
        )

    def _story_choice_point(
        self,
        screen: np.ndarray,
    ) -> tuple[int, int] | None:
        """Return the first story choice while the visible Skip is disabled."""
        behavior = self.config.get("behavior", {})
        if not bool(behavior.get("story_choice_before_skip", True)):
            return None

        raw_roi = behavior.get(
            "story_choice_scan_roi",
            [230, 140, 1370, 560],
        )
        if not isinstance(raw_roi, list) or len(raw_roi) != 4:
            return None
        x1, y1 = self._scaled_point((raw_roi[0], raw_roi[1]), screen)
        x2, y2 = self._scaled_point((raw_roi[2], raw_roi[3]), screen)
        height, width = screen.shape[:2]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        # FGO's choice rows are wide black panels. Text can split a dark row
        # into two short runs, so merge nearby runs before counting choices.
        value = cv2.cvtColor(
            screen[y1:y2, x1:x2],
            cv2.COLOR_BGR2HSV,
        )[:, :, 2]
        dark_value_max = int(
            behavior.get("story_choice_dark_value_max", 48)
        )
        min_dark_ratio = float(
            behavior.get("story_choice_min_dark_ratio", 0.65)
        )
        dark_rows = np.mean(value <= dark_value_max, axis=1) >= min_dark_ratio

        base_height = float(self.config["screen"]["base_height"])
        scale_y = height / base_height
        min_run = max(
            2,
            round(
                float(behavior.get("story_choice_min_dark_run", 15))
                * scale_y
            ),
        )
        merge_gap = max(
            1,
            round(
                float(behavior.get("story_choice_text_gap", 45))
                * scale_y
            ),
        )
        min_choice_height = max(
            min_run,
            round(
                float(behavior.get("story_choice_min_height", 55))
                * scale_y
            ),
        )
        max_choice_height = max(
            min_choice_height,
            round(
                float(behavior.get("story_choice_max_height", 145))
                * scale_y
            ),
        )
        min_choice_gap = max(
            1,
            round(
                float(behavior.get("story_choice_min_center_gap", 80))
                * scale_y
            ),
        )

        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, is_dark in enumerate(dark_rows):
            if is_dark and start is None:
                start = index
            elif not is_dark and start is not None:
                if index - start >= min_run:
                    runs.append((start, index - 1))
                start = None
        if start is not None and len(dark_rows) - start >= min_run:
            runs.append((start, len(dark_rows) - 1))

        # A dark scene background can fill the top or bottom edge of the
        # scan ROI and sit only a few pixels away from a real choice panel.
        # Such an edge-touching run is background, not a complete choice;
        # remove it before the text-gap merge so it cannot inflate the first
        # panel beyond ``story_choice_max_height``.
        runs = [
            (run_start, run_end)
            for run_start, run_end in runs
            if run_start > 0 and run_end < len(dark_rows) - 1
        ]

        merged: list[tuple[int, int]] = []
        for run_start, run_end in runs:
            if merged and run_start - merged[-1][1] - 1 <= merge_gap:
                merged[-1] = (merged[-1][0], run_end)
            else:
                merged.append((run_start, run_end))
        choices = [
            (run_start, run_end)
            for run_start, run_end in merged
            if min_choice_height <= run_end - run_start + 1 <= max_choice_height
        ]
        if len(choices) < 2:
            return None
        first_center = (choices[0][0] + choices[0][1]) // 2
        second_center = (choices[1][0] + choices[1][1]) // 2
        if second_center - first_center < min_choice_gap:
            return None
        return ((x1 + x2) // 2, y1 + first_center)

    def _story_choice_match(
        self,
        screen: np.ndarray,
    ) -> MatchResult | None:
        # 刷本模式不推进剧情：暗色行扫描可能误伤战斗指令卡界面。
        if self.farm_mode:
            return None
        if str(self.config["behavior"].get("story_mode", "skip")) != "skip":
            return None
        point = self._story_choice_point(screen)
        if point is None:
            return None
        return MatchResult(
            rule={
                "name": "story_choice_first",
                "action": "tap_match",
                "cooldown": 2,
                "unknown_pause_seconds": 60,
            },
            score=1.0,
            x=point[0] - 1,
            y=point[1] - 1,
            width=2,
            height=2,
        )

    def _story_visible_prefer(self, screen: np.ndarray) -> list[str]:
        """返回当前画面可见的剧情规则名，供主匹配优先复检。

        跳过确认弹窗上，宽松阈值的 quest_confirm_start（0.7）会先于
        skip_confirm 命中并点击弹窗空白处。此处提前检出可见的剧情
        规则，让 skip_confirm / 跳过按钮优先于其他规则执行。
        """
        if (
            self.farm_mode
            or str(self.config["behavior"].get("story_mode", "skip"))
            != "skip"
        ):
            return []
        preferred: list[str] = []
        # 弹窗优先于跳过按钮：确认弹窗打开时跳过按钮仍在画面右上角。
        for name in (
            "skip_confirm",
            "story_skip_light_background",
            "story_skip",
            "story_skip_gray_text_fallback",
        ):
            rule = self._rules_by_name.get(name)
            if (
                rule is not None
                and self.matcher.match_rule(screen, rule) is not None
            ):
                preferred.append(name)
        return preferred

    def _current_exclude_rules(self) -> set[str]:
        """当前帧的规则排除集：战斗专属规则在非战斗画面不参与匹配。"""
        exclude = set(self._exclude_rules)
        if not self._battle_flow_active:
            exclude |= self._battle_only_rules
        return exclude

    def _tap(self, point: tuple[int, int], label: str) -> None:
        self.log(f"点击 {label}：({point[0]}, {point[1]})")
        if not self.dry_run:
            self.device.tap(*point)

    def _close_free_quest(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> None:
        """Close a free-quest detail strip and verify the map is visible."""
        attempts = max(1, int(match.rule.get("retry_attempts", 3)))
        wait_seconds = max(
            0.1,
            float(match.rule.get("retry_wait_seconds", 0.8)),
        )
        current = screen
        for attempt in range(1, attempts + 1):
            current_match = self._free_quest_match(current)
            if current_match is None:
                self.log("自由关卡详情已经关闭，继续在地图寻找“下一个”")
                return
            self._tap(
                current_match.center,
                f"自由关卡左上角关闭 {attempt}/{attempts}",
            )
            self._wait_interruptibly(wait_seconds)
            if self.stop_event.is_set():
                return
            current = self.device.capture()
            if self._free_quest_match(current) is None:
                self.log("自由关卡详情关闭成功，继续在地图寻找“下一个”")
                return
        self._pause(
            f"连续点击自由关卡“关闭”{attempts}次后界面仍未关闭",
            current,
            self._free_quest_match(current),
        )

    def _tap_match_until_gone(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> None:
        """Retry an idempotent close button and verify its template is gone."""
        rule = match.rule
        attempts = max(1, int(rule.get("retry_attempts", 3)))
        wait_seconds = max(
            0.1,
            float(rule.get("retry_wait_seconds", 0.8)),
        )
        offset = rule.get("offset", [0, 0])
        current = screen
        current_match = match

        for attempt in range(1, attempts + 1):
            scaled_offset = self._scaled_point(offset, current)
            point = (
                current_match.center[0] + scaled_offset[0],
                current_match.center[1] + scaled_offset[1],
            )
            self._tap(
                point,
                f"{rule['name']} {attempt}/{attempts}",
            )
            self._wait_interruptibly(wait_seconds)
            if self.stop_event.is_set():
                return
            current = self.device.capture()
            remaining = self.matcher.match_rule(current, rule)
            if remaining is None:
                self.log(
                    f"{rule['name']} 点击后已消失，关闭成功"
                )
                return
            current_match = remaining
            if attempt < attempts:
                self.log(
                    f"{rule['name']} 点击后仍在画面中，准备重试 "
                    f"{attempt + 1}/{attempts}"
                )

        self._pause(
            f"连续点击 {rule['name']} {attempts} 次后界面仍未关闭",
            current,
            current_match,
        )

    def _pause(
        self,
        reason: str,
        screen: np.ndarray,
        match: MatchResult | None = None,
    ) -> None:
        result = self.selfcheck.try_recover(
            reason,
            screen,
            match,
            in_battle=self._battle_flow_active,
        )
        if result is not None:
            raise RecoveredAction(reason, result)
        path = self._snapshot(screen, "PAUSED", match)
        raise PauseRequested(reason, path)

    def _looks_like_story_screen(self, screen: np.ndarray) -> bool:
        """粗略识别剧情演出画面：中央区域整体偏暗，底部有白色对话文字。

        用于延长未知画面等待（剧情演出期间不盲目点击，等跳过按钮
        出现）。黑屏等纯暗画面没有底部文字，不会被误判。
        """
        behavior = self.config["behavior"]
        height, width = screen.shape[:2]
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        center = hsv[
            round(height * 0.30) : round(height * 0.65),
            round(width * 0.25) : round(width * 0.75),
        ]
        if center.size == 0:
            return False
        if (
            float(center[:, :, 2].mean())
            > float(behavior.get("story_screen_max_center_value", 60))
        ):
            return False
        bottom = hsv[
            round(height * 0.78) : height,
            round(width * 0.10) : round(width * 0.90),
        ]
        if bottom.size == 0:
            return False
        white = (bottom[:, :, 1] <= 110) & (bottom[:, :, 2] >= 165)
        return float(np.count_nonzero(white)) / float(white.size) >= float(
            behavior.get("story_screen_min_bottom_white", 0.005)
        )

    def _unknown_pause_policy(
        self,
        screen: np.ndarray,
    ) -> tuple[float, str, str]:
        if self._battle_flow_active:
            return (
                max(
                    self._current_unknown_limit,
                    self.battle_animation_pause_seconds,
                ),
                "battle_animation",
                "战斗演出或敌方行动画面",
            )
        if (
            self._has_seen_recognized_rule
            and not self.farm_mode
            and self._looks_like_story_screen(screen)
        ):
            return (
                max(
                    self._current_unknown_limit,
                    float(
                        self.config["behavior"].get(
                            "story_unknown_pause_seconds",
                            90,
                        )
                    ),
                ),
                "story",
                "剧情演出画面",
            )
        if not self._has_seen_recognized_rule:
            return (
                max(
                    self._current_unknown_limit,
                    self.startup_unknown_pause_seconds,
                ),
                "startup",
                "启动时加载或演出画面",
            )
        return (
            self._current_unknown_limit,
            "unknown",
            "未知画面",
        )

    def _log_unknown_wait_context(
        self,
        context: str,
        description: str,
        limit: float,
    ) -> None:
        if context == self._unknown_wait_context:
            return
        self._unknown_wait_context = context
        if context == "battle_animation":
            self.log(
                f"未检测到可操作按钮，按{description}等待；"
                f"最长 {limit:.0f} 秒，期间不会盲目点击"
            )
        elif context == "story":
            self.log(
                f"画面疑似{description}，未检测到可操作按钮；"
                f"最长等待 {limit:.0f} 秒，期间不会盲目点击"
            )
        elif context == "startup":
            self.log(
                f"启动后暂未检测到可操作界面，可能处于{description}；"
                f"最长等待 {limit:.0f} 秒"
            )

    def _check_foreground(self, screen: np.ndarray) -> None:
        if not self.expected_package:
            return
        now = time.monotonic()
        interval = float(
            self.config["behavior"].get("foreground_check_seconds", 15)
        )
        if now - self._last_foreground_check < interval:
            return
        self._last_foreground_check = now
        package = self.device.foreground_package()
        if package != self.expected_package:
            self._pause(
                f"前台应用不是 FGO（当前：{package or '未知'}），已停止",
                screen,
            )

    def _wait_interruptibly(self, seconds: float) -> None:
        self.stop_event.wait(max(0.0, seconds))

    def _poll_wait(self) -> None:
        # 等待期间让预取线程并行截取下一帧，掩盖 ADB 截图延迟。
        self._prefetcher.request()
        self._wait_interruptibly(self.poll_seconds)

    def _wait_with_skill_animation_tap(
        self,
        seconds: float,
        screen: np.ndarray,
        prefix: str,
        *,
        start_delay_seconds: float | None = None,
        animation_started_by_previous_tap: bool = False,
    ) -> None:
        """Tap once during a confirmed skill animation without touching modals."""
        total_seconds = max(0.0, float(seconds))
        if (
            total_seconds <= 0
            or self.dry_run
            or self.stop_event.is_set()
            or not self._battle_rule_enabled("accelerate_skill_animations")
        ):
            self._wait_interruptibly(total_seconds)
            return

        battle = self.config.get("battle", {})
        configured_start_delay = (
            battle.get(
                "skill_animation_tap_start_delay_seconds",
                battle.get(
                    "skill_animation_hold_start_delay_seconds",
                    0.20,
                ),
            )
            if start_delay_seconds is None
            else start_delay_seconds
        )
        start_delay = min(
            total_seconds,
            max(0.0, float(configured_start_delay)),
        )
        self._wait_interruptibly(start_delay)
        if self.stop_event.is_set():
            return

        remaining_seconds = max(0.0, total_seconds - start_delay)
        blocking_reason: str | None = None
        if not animation_started_by_previous_tap:
            # The caller passes the fresh probe captured immediately after
            # the skill/confirmation tap. Avoid a second ADB screenshot here:
            # on short animations that extra round trip arrives too late.
            if self.battle_planner.skill_confirmation_visible(screen):
                blocking_reason = "技能使用确认框仍在"
            elif self._skill_target_overlay_visible(screen):
                blocking_reason = "技能选人框仍在"
            elif self._skill_unavailable_overlay_visible(screen):
                blocking_reason = "技能不可用提示仍在"
        if blocking_reason is not None:
            self.log(
                f"{prefix}：{blocking_reason}，跳过本次动画加速点击以避免误触"
            )
            self._wait_interruptibly(remaining_seconds)
            return

        # 加速点击不再使用固定坐标：左上角等位置在不同关卡会落在
        # 敌人/从者身上，误触后弹出人物状态界面拖慢流程。改为现场
        # 截取一帧，在画面上寻找平坦的背景补丁再点击；找不到就跳过。
        current = screen
        try:
            current = self.device.capture()
        except AdbError:
            current = screen
        if self.stop_event.is_set():
            return
        point = self._animation_safe_point(current)
        if point is None:
            self.log(
                f"{prefix}：技能动画画面未找到足够平坦的安全区域，"
                "跳过本次加速点击（避免误触人物或按钮）"
            )
            self._wait_interruptibly(remaining_seconds)
            return
        self.log(
            f"{prefix}：技能动画已进入无弹窗阶段，"
            f"单击安全位置 ({point[0]}, {point[1]}) 加速一次"
        )
        self._tap(point, "技能动画单击加速")
        # Preserve the original post-skill wait. Acceleration does not replace
        # synchronization with the stable Attack screen.
        self._wait_interruptibly(remaining_seconds)

    def _animation_safe_point(
        self,
        screen: np.ndarray,
    ) -> tuple[int, int] | None:
        """在当前帧寻找平坦的背景补丁作为技能动画加速点击的安全位置。

        固定坐标在不同关卡会落在敌人/从者身上（误触弹出人物状态界面），
        因此扫描候选网格：补丁需满足低边缘密度、低亮度波动、中低亮度、
        低饱和，且 3x3 分块均一（拒绝"角色边缘+背景"的混合补丁）。
        找不到足够平坦的区域时返回 None，调用方跳过点击。
        """
        battle = self.config.get("battle", {})
        height, width = screen.shape[:2]
        base_width = float(self.config["screen"]["base_width"])
        base_height = float(self.config["screen"]["base_height"])
        scale_x = width / base_width
        scale_y = height / base_height
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 60, 150)
        patch_size = max(
            24,
            round(
                float(battle.get("animation_tap_patch_px", 60))
                * min(scale_x, scale_y)
            ),
        )
        max_edge_ratio = float(
            battle.get("animation_tap_max_edge_ratio", 0.10)
        )
        max_stddev = float(battle.get("animation_tap_max_stddev", 26.0))
        min_value = float(battle.get("animation_tap_min_value", 25.0))
        max_value = float(battle.get("animation_tap_max_value", 150.0))
        max_saturation = float(
            battle.get("animation_tap_max_saturation", 70.0)
        )
        best: tuple[float, float, int, int] | None = None
        # 网格覆盖战场中部：避开顶部血条区与底部指令卡区。
        inner_half = max(8, patch_size // 4)
        for raw_y in range(150, 570, 60):
            for raw_x in range(150, 1450, 100):
                cx = min(width - 1, round(raw_x * scale_x))
                cy = min(height - 1, round(raw_y * scale_y))
                # 点击只会落在补丁中心，因此只对中心附近的小区域做
                # 平坦性判定：角色/按钮再近，只要中心是背景就安全。
                x1, y1 = max(0, cx - inner_half), max(0, cy - inner_half)
                x2, y2 = min(width, cx + inner_half), min(
                    height,
                    cy + inner_half,
                )
                if x2 - x1 < 12 or y2 - y1 < 12:
                    continue
                inner_gray = gray[y1:y2, x1:x2]
                inner_hsv = hsv[y1:y2, x1:x2]
                mean_v = float(inner_hsv[:, :, 2].mean())
                if not (min_value <= mean_v <= max_value):
                    continue
                if float(inner_hsv[:, :, 1].mean()) > max_saturation:
                    continue
                stddev = float(inner_gray.std())
                if stddev > max_stddev:
                    continue
                edge_ratio = (
                    float(np.count_nonzero(edges[y1:y2, x1:x2]))
                    / float(inner_gray.size)
                )
                if edge_ratio > max_edge_ratio:
                    continue
                if best is None or (edge_ratio, stddev) < (
                    best[0],
                    best[1],
                ):
                    best = (edge_ratio, stddev, cx, cy)
        if best is None:
            return None
        return best[2], best[3]

    def _skill_target_overlay_visible(self, screen: np.ndarray) -> bool:
        target_rule = next(
            (
                rule
                for rule in self.config.get("rules", [])
                if rule.get("action") == "target_damage_dealer"
            ),
            None,
        )
        match_rule = getattr(self.matcher, "match_rule", None)
        if target_rule is not None and callable(match_rule):
            recovery_rule = dict(target_rule)
            recovery_rule["threshold"] = min(
                float(target_rule.get("threshold", 0.9)),
                float(
                    self.config.get("battle", {}).get(
                        "skill_target_recovery_threshold",
                        0.70,
                    )
                ),
            )
            if match_rule(screen, recovery_rule) is not None:
                return True
        # 模板未命中时用布局启发式兜底：选人界面会遮住攻击按钮、
        # 也没有技能确认弹窗，同时三张选人头像齐全。三者同时成立
        # 是选人界面的特征组合（实测模板在此界面上只有约 0.36 分）。
        if (
            len(self.battle_planner.skill_target_points(screen)) >= 3
            and not self.battle_planner.skill_confirmation_visible(screen)
            and match_rule is not None
        ):
            attack_rule = next(
                (
                    rule
                    for rule in self.config.get("rules", [])
                    if rule.get("name") == "battle_attack"
                ),
                None,
            )
            attack_match = (
                match_rule(screen, attack_rule)
                if attack_rule is not None
                else None
            )
            if attack_match is None:
                return True
        return False

    def _skill_unavailable_overlay_visible(
        self,
        screen: np.ndarray,
    ) -> bool:
        """Return True when a failed skill has opened its Close dialog."""
        close_rule = next(
            (
                rule
                for rule in self.config.get("rules", [])
                if rule.get("name") == "skill_unavailable_close"
            ),
            None,
        )
        match_rule = getattr(self.matcher, "match_rule", None)
        return bool(
            close_rule is not None
            and callable(match_rule)
            and match_rule(screen, close_rule) is not None
        )

    def _resolve_pending_skill_target(
        self,
        screen: np.ndarray,
        prefix: str,
        *,
        target_slot: int | None = None,
        active_slots: list[int] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Select a pending skill target before Attack can touch the modal."""
        if not self._skill_target_overlay_visible(screen):
            return screen, False

        current = screen
        if target_slot is None or active_slots is None:
            context = self._battle_skill_context(current)
            target_slot = int(context["damage_slot"])
            active_slots = list(context["active_slots"])

        attempts = max(
            1,
            int(
                self.config.get("battle", {}).get(
                    "skill_target_select_attempts",
                    2,
                )
            ),
        )
        wait_seconds = float(
            self.config.get("battle", {}).get(
                "after_skill_target_wait_seconds",
                1.2,
            )
        )
        for attempt in range(1, attempts + 1):
            point = self._skill_target_point(
                current,
                int(target_slot),
                list(active_slots),
            )
            suffix = "" if attempt == 1 else f"（重试{attempt}）"
            self._tap(
                point,
                f"{prefix}选择当前输出位置{target_slot}{suffix}",
            )
            self._wait_with_skill_animation_tap(
                wait_seconds,
                current,
                f"{prefix}选人后",
                start_delay_seconds=float(
                    self.config.get("battle", {}).get(
                        "skill_animation_after_target_tap_delay_seconds",
                        0.08,
                    )
                ),
                animation_started_by_previous_tap=True,
            )
            if self.stop_event.is_set():
                return current, True
            current = self.device.capture()
            if not self._skill_target_overlay_visible(current):
                return current, True
            self.log(f"{prefix}选人界面仍在，准备重新选择目标")

        self._pause(
            f"{prefix}连续{attempts}次选择人物后界面仍未关闭；"
            "为防止旧Attack坐标误触退出，已暂停",
            current,
        )
        raise AssertionError("unreachable")

    def _skill_cancel_point(self) -> tuple[float, float]:
        """技能确认弹窗“返回”按钮中心（基准坐标）。"""
        battle = self.config.get("battle", {})
        raw = battle.get("skill_cancel_button_region", [325, 480, 735, 575])
        return (
            (float(raw[0]) + float(raw[2])) / 2.0,
            (float(raw[1]) + float(raw[3])) / 2.0,
        )

    def _resolve_skill_confirmation(
        self,
        screen: np.ndarray,
        prefix: str,
        *,
        accelerate_animation: bool,
    ) -> tuple[np.ndarray, bool]:
        """Close a visible skill confirmation before animation acceleration."""
        if (
            self.dry_run
            or self.stop_event.is_set()
            or not self._battle_rule_enabled("confirm_skill_use")
            or not self.battle_planner.skill_confirmation_visible(screen)
        ):
            return screen, False

        battle = self.config.get("battle", {})
        current = screen
        # 弹窗关闭动画约 0.3~0.5 秒。探测等待太短会在关闭动画期间
        # 误判“点击无效”，补点的“决定”还会点到战场（看起来就是
        # 点击位置不对且画面反复不推进）。强制最短 0.35 秒。
        probe_seconds = max(
            0.35,
            float(
                battle.get(
                    "skill_confirmation_after_tap_probe_seconds",
                    0.35,
                )
            ),
        )
        attempts = max(
            1,
            int(battle.get("skill_confirmation_max_attempts", 2)),
        )
        skill_used = True
        for attempt in range(1, attempts + 1):
            confirm_point = self._scaled_point(
                battle.get("skill_confirm_point", [1050, 530]),
                current,
            )
            suffix = "" if attempt == 1 else f"（重试{attempt}）"
            self._tap(confirm_point, f"{prefix}确认技能使用{suffix}")
            self._wait_interruptibly(probe_seconds)
            if self.stop_event.is_set():
                return current, True
            current = self.device.capture()
            if not self.battle_planner.skill_confirmation_visible(current):
                break
            self.log(f"{prefix}技能确认框仍在，准备重试“决定”")
        else:
            # “决定”点了多次弹窗仍不关闭：技能此刻可能不可用，
            # 继续点只会死循环。分析画面后改点“返回”关掉弹窗，
            # 让战斗继续推进，而不是停在原地等待人工。
            skill_used = False
            cancel_point = self._scaled_point(
                self._skill_cancel_point(),
                current,
            )
            self.log(
                f"{prefix}“决定”点击{attempts}次未关闭弹窗，"
                f"改为点击“返回”({cancel_point[0]}, {cancel_point[1]})"
                "关闭后继续战斗"
            )
            self._tap(cancel_point, "技能弹窗返回")
            self._wait_interruptibly(probe_seconds)
            if self.stop_event.is_set():
                return current, True
            current = self.device.capture()
            if self.battle_planner.skill_confirmation_visible(current):
                self._pause(
                    f"{prefix}技能确认框点击“决定”{attempts}次、"
                    "“返回”1次仍未关闭",
                    current,
                )
            self.log(f"{prefix}技能弹窗已通过“返回”关闭")

        post_wait_seconds = max(
            0.0,
            float(battle.get("after_skill_confirm_wait_seconds", 1.0))
            - probe_seconds,
        )
        if self._skill_target_overlay_visible(current):
            # A directed skill has only reached its target-selection screen.
            # Tapping here could interfere with selecting a servant.
            self._wait_interruptibly(post_wait_seconds)
        elif accelerate_animation and skill_used:
            self._wait_with_skill_animation_tap(
                post_wait_seconds,
                current,
                f"{prefix}确认后",
                start_delay_seconds=0,
            )
        else:
            # 通过“返回”关闭弹窗时技能没有释放，不存在演出动画，
            # 只做普通等待，避免加速点击点到战场。
            self._wait_interruptibly(post_wait_seconds)
        if not self.stop_event.is_set():
            current = self.device.capture()
        return current, True

    def _wait_until_battle_ready_after_skill(
        self,
        screen: np.ndarray,
        prefix: str,
        *,
        target_slot: int | None = None,
        active_slots: list[int] | None = None,
    ) -> np.ndarray:
        """Wait for a real, stable Attack screen after using a skill.

        Skill animations can leave the lower battle HUD visible while bright
        effects cover the fixed skill coordinates.  Evaluating cooldowns on
        such a frame makes every skill look ready and can also send an Attack
        tap into the animation.  Synchronize on the actual Attack template,
        while still recovering modals that may appear a little late.
        """
        match_rule = getattr(self.matcher, "match_rule", None)
        attack_rule = next(
            (
                rule
                for rule in self.config.get("rules", [])
                if rule.get("name") == "battle_attack"
            ),
            None,
        )
        if attack_rule is None or not callable(match_rule) or self.dry_run:
            return screen

        battle = self.config.get("battle", {})
        timeout_seconds = max(
            1.0,
            float(battle.get("post_skill_ready_timeout_seconds", 18.0)),
        )
        poll_seconds = max(
            0.1,
            float(battle.get("post_skill_ready_poll_seconds", 0.35)),
        )
        stable_required = max(
            1,
            int(battle.get("post_skill_ready_stable_frames", 2)),
        )
        skill_error_rule = next(
            (
                rule
                for rule in self.config.get("rules", [])
                if rule.get("name") == "skill_unavailable_close"
            ),
            None,
        )
        deadline = time.monotonic() + timeout_seconds
        stable_frames = 0
        current = screen
        wait_logged = False

        while not self.stop_event.is_set():
            current, confirmed = self._resolve_skill_confirmation(
                current,
                f"{prefix}·延迟确认",
                accelerate_animation=True,
            )
            if confirmed:
                stable_frames = 0

            skill_error_match = (
                match_rule(current, skill_error_rule)
                if skill_error_rule is not None
                else None
            )
            if skill_error_match is not None:
                self.log(f"{prefix}：检测到延迟出现的技能不可用提示，自动关闭")
                self._tap(
                    skill_error_match.center,
                    "关闭延迟出现的技能不可用提示",
                )
                self._wait_interruptibly(
                    float(battle.get("skill_error_close_wait_seconds", 1.0))
                )
                if self.stop_event.is_set():
                    return current
                current = self.device.capture()
                stable_frames = 0
                continue

            current, resolved_target = self._resolve_pending_skill_target(
                current,
                f"{prefix}·延迟选人",
                target_slot=target_slot,
                active_slots=active_slots,
            )
            if resolved_target:
                stable_frames = 0

            attack_match = match_rule(current, attack_rule)
            if attack_match is not None:
                stable_frames += 1
                if stable_frames >= stable_required:
                    self.log(
                        f"{prefix}：技能动画结束，已连续 {stable_frames} 帧确认攻击界面"
                    )
                    return current
            else:
                stable_frames = 0

            if time.monotonic() >= deadline:
                self._pause(
                    f"{prefix}后等待技能动画结束 {timeout_seconds:.1f} 秒，"
                    "仍未恢复攻击界面",
                    current,
                )

            if not wait_logged:
                self.log(
                    f"{prefix}：等待技能动画结束并恢复攻击界面，"
                    "期间不再判断技能或点击攻击"
                )
                wait_logged = True
            self._wait_interruptibly(poll_seconds)
            if not self.stop_event.is_set():
                current = self.device.capture()

        return current

    def _screen_fingerprint(self, screen: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        return cv2.resize(
            gray,
            (48, 27),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

    def _fingerprint_similar(
        self,
        a: np.ndarray | None,
        b: np.ndarray | None,
        tolerance: float | None = None,
    ) -> bool:
        """48×27 指纹近似比较：均值绝对差小于容差视为同一画面。

        用于未知画面：忽略时钟、粒子等微小扰动，避免无意义的全量重匹配。
        """
        if a is None or b is None or a.shape != b.shape:
            return False
        if tolerance is None:
            tolerance = float(
                self.config["behavior"].get(
                    "fingerprint_similar_tolerance",
                    1.5,
                )
            )
        return float(np.mean(np.abs(a - b))) < tolerance

    def _unknown_static_limit(self, context: str) -> float | None:
        """未知画面静止 N 秒直接触发自检，不必等满超时。

        startup 返回 None（冷启动画面静止是正常的，保持 120 秒宽限）；
        战斗演出画面大多在动，静止阈值放宽；其余画面 10 秒。
        """
        behavior = self.config["behavior"]
        if context == "startup":
            return None
        if context == "battle_animation":
            return float(
                behavior.get("unknown_static_pause_seconds_battle", 20.0)
            )
        if context == "story":
            return float(
                behavior.get("unknown_static_pause_seconds_story", 25.0)
            )
        return float(
            behavior.get("unknown_static_pause_seconds", 10.0)
        )

    def _reset_unknown_static(self) -> None:
        self._unknown_static_since = 0.0
        self._unknown_static_fingerprint = None
        self._static_skip_count = 0

    def _support_list_visual_change(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> float:
        raw_region = self.config.get("support", {}).get(
            "scroll_list_region",
            [260, 210, 1510, 875],
        )
        base = self.config["screen"]

        def crop(screen: np.ndarray) -> np.ndarray:
            height, width = screen.shape[:2]
            x1 = max(
                0,
                round(
                    float(raw_region[0])
                    * width
                    / float(base["base_width"])
                ),
            )
            y1 = max(
                0,
                round(
                    float(raw_region[1])
                    * height
                    / float(base["base_height"])
                ),
            )
            x2 = min(
                width,
                round(
                    float(raw_region[2])
                    * width
                    / float(base["base_width"])
                ),
            )
            y2 = min(
                height,
                round(
                    float(raw_region[3])
                    * height
                    / float(base["base_height"])
                ),
            )
            return screen[y1:y2, x1:x2]

        before_roi = crop(before)
        after_roi = crop(after)
        if before_roi.size == 0 or after_roi.size == 0:
            return 0.0
        before_fingerprint = self._screen_fingerprint(before_roi)
        after_fingerprint = self._screen_fingerprint(after_roi)
        return float(
            np.mean(
                np.abs(after_fingerprint - before_fingerprint)
            )
        )

    def _support_scroll_step(
        self,
        screen: np.ndarray,
        settings: dict[str, Any],
    ) -> tuple[int, int | None]:
        """Use an overlapping drag so edge-clipped rows reappear fully."""
        configured_step = float(settings.get("scroll_step_pixels", 340))
        rows_per_scroll = max(
            1,
            int(settings.get("scroll_rows_per_page", 2)),
        )
        row_drag_ratio = max(
            0.1,
            min(1.0, float(settings.get("scroll_row_drag_ratio", 0.68))),
        )
        row_method = getattr(
            self.support_selector,
            "normal_support_row_ranges",
            None,
        )
        row_pitch: int | None = None
        if callable(row_method):
            rows = row_method(screen)
            row_tops = sorted(int(top) for top, _bottom in rows)
            gaps = [
                second - first
                for first, second in zip(row_tops, row_tops[1:])
                if 180 <= second - first <= 320
            ]
            if gaps:
                row_pitch = round(float(np.median(gaps)))
                # FGO adds momentum after the finger is released.  Dragging a
                # full two row pitches moves nearly three rows on some lists,
                # leaving one candidate clipped before and after the swipe.
                configured_step = float(
                    row_pitch * rows_per_scroll * row_drag_ratio
                )

        scale_y = screen.shape[0] / float(
            self.config["screen"]["base_height"]
        )
        minimum_step = round(
            float(settings.get("scroll_step_min_pixels", 1)) * scale_y
        )
        maximum_step = round(
            float(
                settings.get(
                    "scroll_step_max_pixels",
                    self.config["screen"]["base_height"],
                )
            )
            * scale_y
        )
        raw_step = (
            round(configured_step)
            if row_pitch is not None
            else round(configured_step * scale_y)
        )
        step = max(minimum_step, min(maximum_step, raw_step))
        return max(1, step), row_pitch

    def _clear_blocked_rule(self) -> None:
        self._blocked_rule = None
        self._blocked_fingerprint = None
        self._blocked_center = None

    def _blocked_screen_has_changed(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> tuple[bool, float, float]:
        if self._blocked_fingerprint is None:
            return False, 0.0, 0.0
        current = self._screen_fingerprint(screen)
        visual_change = float(
            np.mean(np.abs(current - self._blocked_fingerprint))
        )
        center_change = 0.0
        if self._blocked_center is not None:
            dx = match.center[0] - self._blocked_center[0]
            dy = match.center[1] - self._blocked_center[1]
            center_change = float((dx * dx + dy * dy) ** 0.5)
        behavior = self.config["behavior"]
        return (
            visual_change
            >= float(behavior.get("same_rule_screen_change_threshold", 8.0))
            or center_change
            >= float(behavior.get("same_rule_center_change_threshold", 35.0)),
            visual_change,
            center_change,
        )

    def _perform_steps(
        self,
        steps: list[dict[str, Any]],
        screen: np.ndarray,
        prefix: str,
        *,
        confirm_skills: bool = False,
        accelerate_skill_animation: bool = False,
    ) -> np.ndarray:
        current = screen
        target_reference = screen
        for index, step in enumerate(steps, start=1):
            if self.stop_event.is_set():
                return current
            if "wait" in step:
                self._wait_interruptibly(float(step["wait"]))
                continue
            if "tap" in step:
                point = self._scaled_point(step["tap"], current)
            elif "tap_norm" in step:
                height, width = current.shape[:2]
                point = (
                    round(float(step["tap_norm"][0]) * width),
                    round(float(step["tap_norm"][1]) * height),
                )
            elif step.get("tap_damage_dealer"):
                if not self.dry_run:
                    current = self.device.capture()
                    if not self._skill_target_overlay_visible(current):
                        self.log(
                            f"{prefix}第{index}步未检测到技能选人框；"
                            "禁止点击人物区域，跳过本次选人动作"
                        )
                        continue
                legacy_point, slot = self._damage_dealer_point(
                    target_reference
                )
                active_slots = self.battle_planner.active_party_slots(
                    target_reference
                )
                if self.dry_run:
                    point = legacy_point
                else:
                    point = self._skill_target_point(
                        current,
                        slot,
                        active_slots,
                    )
                self.log(f"{prefix}第{index}步选择主力输出位 {slot}")
            else:
                raise PauseRequested(
                    f"{prefix} 第 {index} 步没有 tap/tap_norm/"
                    "tap_damage_dealer/wait"
                )
            self._tap(point, f"{prefix}步骤{index}")
            after_seconds = float(step.get("after", 0.55))
            next_action = next(
                (
                    candidate
                    for candidate in steps[index:]
                    if "wait" not in candidate
                ),
                None,
            )
            opens_target_selection = bool(
                next_action is not None
                and next_action.get("tap_damage_dealer")
            )
            starts_skill_animation = (
                bool(step.get("tap_damage_dealer"))
                or (
                    ("tap" in step or "tap_norm" in step)
                    and not opens_target_selection
                )
            )
            can_probe_confirmation = (
                confirm_skills
                and "tap" in step
                and bool(step.get("confirm_skill", True))
                and not self.dry_run
                and not self.stop_event.is_set()
            )
            if can_probe_confirmation:
                probe_seconds = min(
                    after_seconds,
                    max(
                        0.0,
                        float(
                            self.config.get("battle", {}).get(
                                "skill_confirmation_probe_seconds",
                                0.8,
                            )
                        ),
                    ),
                )
                self._wait_interruptibly(probe_seconds)
                if self.stop_event.is_set():
                    return current
                current = self.device.capture()
                current, confirmed = self._resolve_skill_confirmation(
                    current,
                    prefix,
                    accelerate_animation=(
                        accelerate_skill_animation
                        and not opens_target_selection
                    ),
                )
                if not confirmed:
                    remaining_seconds = max(
                        0.0,
                        after_seconds - probe_seconds,
                    )
                    if (
                        accelerate_skill_animation
                        and starts_skill_animation
                        and not self._skill_target_overlay_visible(current)
                    ):
                        self._wait_with_skill_animation_tap(
                            remaining_seconds,
                            current,
                            f"{prefix}第{index}步",
                            start_delay_seconds=0,
                        )
                    else:
                        self._wait_interruptibly(remaining_seconds)
            elif (
                accelerate_skill_animation
                and bool(step.get("tap_damage_dealer"))
            ):
                # The target has just been selected, so animation has truly
                # started. A very short input-settle delay is enough; taking
                # another screenshot here would miss short animations.
                self._wait_with_skill_animation_tap(
                    after_seconds,
                    current,
                    f"{prefix}第{index}步选人后",
                    start_delay_seconds=float(
                        self.config.get("battle", {}).get(
                            "skill_animation_after_target_tap_delay_seconds",
                            0.08,
                        )
                    ),
                    animation_started_by_previous_tap=True,
                )
            else:
                # If automatic confirmation handling is disabled, never tap
                # after the initial skill tap: a confirmation or target modal
                # could be on top of the battlefield.
                self._wait_interruptibly(after_seconds)
        return current

    def _reset_party_state(self) -> None:
        self._party_state_initialized = False
        self._party_active_slots.clear()
        self._party_identity_features.clear()
        self._party_generations.clear()
        self._party_inactive_slots.clear()
        reset_enemy_layout = getattr(
            self.battle_planner,
            "reset_enemy_layout",
            None,
        )
        if callable(reset_enemy_layout):
            reset_enemy_layout()

    def _sync_party_state(
        self,
        screen: np.ndarray,
    ) -> dict[str, Any]:
        """Track the actual occupant of each visible battle position."""
        active_slots = self.battle_planner.active_party_slots(screen)
        features = self.battle_planner.party_identity_features(
            screen,
            active_slots,
        )
        threshold = float(
            self.config.get("battle", {}).get(
                "party_identity_same_threshold",
                0.94,
            )
        )

        if not self._party_state_initialized:
            self._party_state_initialized = True
            self._party_active_slots = set(active_slots)
            expected_initial_count = int(
                self.config.get("battle", {}).get(
                    "expected_initial_party_count",
                    3,
                )
            )
            joined_midbattle = len(active_slots) < expected_initial_count
            for slot in active_slots:
                self._party_generations[slot] = 1 if joined_midbattle else 0
                if slot in features:
                    self._party_identity_features[slot] = features[slot]
            if joined_midbattle and active_slots:
                self.log(
                    "首次接管时前排已经减员；不套用开局角色身份，"
                    "直接按当前在场从者执行替补策略"
                )
        else:
            no_longer_active = self._party_active_slots - set(active_slots)
            self._party_inactive_slots.update(no_longer_active)
            for slot in active_slots:
                previous = self._party_identity_features.get(slot)
                current = features.get(slot)
                changed = False
                if slot in self._party_inactive_slots:
                    changed = True
                    self._party_inactive_slots.discard(slot)
                elif slot not in self._party_generations:
                    changed = True
                elif previous is not None and current is not None:
                    similarity = float(np.dot(previous, current))
                    changed = similarity < threshold
                if changed:
                    generation = self._party_generations.get(slot, 0) + 1
                    self._party_generations[slot] = generation
                    self.log(
                        f"检测到位置{slot}角色已更换，"
                        f"按当前替补角色重新建立战斗状态（第{generation}次换人）"
                    )
                if current is not None:
                    self._party_identity_features[slot] = current
            self._party_active_slots = set(active_slots)

        initial_slots = [
            slot
            for slot in active_slots
            if self._party_generations.get(slot, 1) == 0
        ]
        replaced_slots = [
            slot
            for slot in active_slots
            if self._party_generations.get(slot, 0) > 0
        ]
        return {
            "active_slots": active_slots,
            "initial_occupant_slots": initial_slots,
            "replaced_slots": replaced_slots,
            "party_generations": dict(self._party_generations),
        }

    def _party_slot_is_initial(self, slot: int) -> bool:
        if not self._party_state_initialized:
            return True
        return (
            slot in self._party_active_slots
            and self._party_generations.get(slot, 1) == 0
        )

    def _damage_dealer_point(
        self,
        screen: np.ndarray,
    ) -> tuple[tuple[int, int], int]:
        battle = self.config.get("battle", {})
        points = battle.get(
            "party_target_points",
            {
                "1": [350, 470],
                "2": [800, 470],
                "3": [1250, 470],
            },
        )
        party_state = self._sync_party_state(screen)
        active_slots = party_state["active_slots"]
        if not active_slots:
            raise PauseRequested(
                "未检测到任何在场从者，已停止选取技能目标，"
                "避免继续操作已经退场的空位置"
            )
        initial_slots = set(party_state["initial_occupant_slots"])
        preferred: list[int] = []
        if self.selected_support is not None:
            support_slot = int(
                self.selected_support.candidate.get(
                    "party_slot",
                    battle.get("primary_damage_slot", 3),
                )
            )
            if support_slot in initial_slots:
                preferred.append(support_slot)
        primary_slot = int(battle.get("primary_damage_slot", 3))
        if primary_slot in initial_slots:
            preferred.append(primary_slot)

        configured_priority = [
            int(value)
            for value in battle.get("damage_slot_priority", [3, 1, 2])
        ]
        priority_index = {
            slot: index for index, slot in enumerate(configured_priority)
        }
        dynamic_slots: list[tuple[int, bool, int, int]] = []
        for active_slot in active_slots:
            value, _ = self.battle_planner.np_charge_percent(
                screen,
                active_slot,
            )
            dynamic_slots.append(
                (
                    active_slot,
                    self._slot_is_damage_role(active_slot),
                    value if value is not None else -1,
                    -priority_index.get(active_slot, 99),
                )
            )
        dynamic_slots.sort(
            key=lambda item: (
                item[1],
                item[2] >= 100,
                item[2],
                item[3],
            ),
            reverse=True,
        )
        preferred.extend(slot for slot, _, _, _ in dynamic_slots)
        slot = next(
            (
                value
                for value in preferred
                if value in active_slots
                and (str(value) in points or value in points)
            ),
            active_slots[0],
        )
        raw_point = points.get(str(slot)) or points.get(slot)
        if raw_point is None:
            raise PauseRequested(f"没有配置主力输出位 {slot} 的选人坐标")
        return self._scaled_point(raw_point, screen), slot

    def _skill_target_point(
        self,
        screen: np.ndarray,
        target_slot: int,
        active_slots: list[int],
    ) -> tuple[int, int]:
        detected_points = self.battle_planner.skill_target_points(screen)
        effective_slots = sorted({int(slot) for slot in active_slots})
        if len(detected_points) == 1:
            return detected_points[0]
        if detected_points and len(effective_slots) != len(detected_points):
            # This method is only called after the high-confidence
            # skill_target_support rule has confirmed “请选择对象”.  On that
            # overlay, three rendered portraits are authoritative: a servant
            # at exactly 1 HP may have no measurable coloured HP run.
            if len(detected_points) == 3:
                live_slots = [1, 2, 3]
            else:
                live_slots = self.battle_planner.active_party_slots(screen)
            if len(live_slots) == len(detected_points):
                self.log(
                    "技能选人界面已按当前画面校正在场位置："
                    f"{effective_slots} → {live_slots}"
                )
                effective_slots = live_slots
                if self._party_state_initialized:
                    self._party_active_slots = set(live_slots)
        if target_slot not in effective_slots and effective_slots:
            priority = [
                int(value)
                for value in self.config.get("battle", {}).get(
                    "damage_slot_priority",
                    [3, 1, 2],
                )
            ]
            fallback_slot = next(
                (
                    slot
                    for slot in priority
                    if slot in effective_slots
                ),
                effective_slots[0],
            )
            self.log(
                f"原输出位 {target_slot} 已不在场，"
                f"技能改为选择当前输出位 {fallback_slot}"
            )
            target_slot = fallback_slot
            self._last_damage_slot = fallback_slot
        point = self.battle_planner.skill_target_point(
            screen,
            target_slot,
            effective_slots,
        )
        if point is not None:
            return point
        detected_count = len(
            detected_points
        )
        self._pause(
            "技能选人界面目标布局无法安全映射："
            f"检测到{detected_count}个头像，"
            f"当前在场位置为{effective_slots}，"
            f"准备选择位置{target_slot}",
            screen,
        )
        raise AssertionError("unreachable")

    def _battle_rule_enabled(self, name: str) -> bool:
        default_enabled = name not in {
            "accelerate_skill_animations",
            "prefer_arts_chain",
        }
        return bool(
            self.config.get("battle", {})
            .get("rule_options", {})
            .get(name, default_enabled)
        )

    def _command_card_recovery_match(
        self,
        screen: np.ndarray,
    ) -> MatchResult | None:
        """Recognize a weak command-card frame before Attack can hit Back."""
        command_rule = next(
            (
                rule
                for rule in self.config.get("rules", [])
                if rule.get("name") == "command_cards"
            ),
            None,
        )
        match_rule = getattr(self.matcher, "match_rule", None)
        if command_rule is None or not callable(match_rule):
            return None
        relaxed = dict(command_rule)
        relaxed["threshold"] = float(
            self.config.get("battle", {}).get(
                "command_card_recovery_threshold",
                0.70,
            )
        )
        recovered = match_rule(screen, relaxed)
        if recovered is None:
            return None
        return MatchResult(
            rule=command_rule,
            score=recovered.score,
            x=recovered.x,
            y=recovered.y,
            width=recovered.width,
            height=recovered.height,
        )

    def _support_rule_enabled(self, name: str) -> bool:
        return bool(
            self.config.get("support", {})
            .get("strategy_options", {})
            .get(name, True)
        )

    def _select_highest_level_enabled(self) -> bool:
        """Whether the override mode "pick the highest level row" is on."""
        return bool(
            self.config.get("support", {})
            .get("strategy_options", {})
            .get("select_highest_level_np", False)
        )

    def _reset_support_class_search(
        self,
        *,
        reset_initial_page: bool = True,
    ) -> None:
        self.support_scrolls = 0
        self.support_scroll_stalls = 0
        self.support_filter_class = None
        self.support_class_queue = []
        self.support_search_levels = []
        self.support_search_rule_keys = []
        self.support_class_index = 0
        self.support_recommended_classes = []
        self.support_search_phase = "strict"
        self.support_candidate_pass = ""
        self.support_default_filter_class = None
        self.support_default_guest_seen = False
        self.support_default_normal_seen = False
        self.support_default_scan_complete = False
        self.support_berserker_guest_seen = False
        self.support_berserker_normal_seen = False
        self.support_berserker_scan_complete = False
        if reset_initial_page:
            self.support_initial_page_scanned = False

    def _damage_dealer_class(
        self,
        screen: np.ndarray,
        damage_slot: int,
    ) -> str | None:
        if not self._party_slot_is_initial(damage_slot):
            return None
        if self.selected_support is not None:
            candidate = self.selected_support.candidate
            candidate_slot = int(
                candidate.get(
                    "party_slot",
                    self.config.get("battle", {}).get("support_slot", 3),
                )
            )
            if damage_slot == candidate_slot:
                value = str(candidate.get("class", "")).strip()
                if value:
                    return value
        classes = self.config.get("battle", {}).get(
            "damage_slot_classes",
            {},
        )
        value = classes.get(str(damage_slot)) or classes.get(damage_slot)
        return str(value).strip() if value else None

    def _slot_is_damage_role(self, slot: int) -> bool:
        """Return whether the current slot is configured as an attacker."""
        battle = self.config.get("battle", {})
        if self.selected_support is not None:
            candidate = self.selected_support.candidate
            candidate_slot = int(
                candidate.get("party_slot", battle.get("support_slot", 3))
            )
            if slot == candidate_slot and self._party_slot_is_initial(slot):
                target = str(
                    candidate.get("np_target", "unknown")
                ).lower()
                return bool(
                    candidate.get("np_damage", target != "support")
                )
        if self._party_slot_is_initial(slot):
            profile = battle.get("np_profiles", {}).get(str(slot), {})
        else:
            profile = battle.get("fallback_np_profile", {})
        target = str(profile.get("target", "single")).lower()
        return bool(profile.get("damage", target != "support"))

    def _battle_skill_context(self, screen: np.ndarray) -> dict[str, Any]:
        party_state = self._sync_party_state(screen)
        active_slots = party_state["active_slots"]
        initial_occupant_slots = party_state["initial_occupant_slots"]
        replaced_slots = party_state["replaced_slots"]
        np_percent_by_slot: dict[int, int | None] = {}
        for slot in active_slots:
            value, _ = self.battle_planner.np_charge_percent(screen, slot)
            np_percent_by_slot[slot] = value
        party_hp_ratios = self.battle_planner.party_health_ratios(screen)
        primary_damage_slot = int(
            self.config.get("battle", {}).get("primary_damage_slot", 3)
        )
        _, damage_slot = self._damage_dealer_point(screen)
        self._last_damage_slot = damage_slot
        damage_np_percent, damage_np_source = (
            self.battle_planner.np_charge_percent(screen, damage_slot)
        )
        if damage_slot not in active_slots:
            damage_np_percent = None
            damage_np_source = "无在场读数"
        charge_threshold = int(
            self.config.get("battle", {}).get(
                "charge_skill_threshold_percent",
                100,
            )
        )
        damage_np_ready = (
            damage_np_percent is not None and damage_np_percent >= 100
        )
        damage_np_needs_charge = (
            damage_np_percent is None
            or damage_np_percent < charge_threshold
        )
        alive_enemies = self.battle_planner.alive_enemy_slots(screen)
        charge_scores = self.battle_planner.enemy_charge_scores(screen)
        max_enemy_charge = max(
            (charge_scores.get(slot, 0.0) for slot in alive_enemies),
            default=0.0,
        )
        return {
            "turn": self.battle_turn,
            "active_slots": active_slots,
            "initial_occupant_slots": initial_occupant_slots,
            "replaced_slots": replaced_slots,
            "party_generations": party_state["party_generations"],
            "active_party_count": len(active_slots),
            "np_percent_by_slot": np_percent_by_slot,
            "party_hp_ratios": party_hp_ratios,
            "min_party_hp_ratio": min(
                (
                    party_hp_ratios.get(slot, 1.0)
                    for slot in active_slots
                ),
                default=1.0,
            ),
            "damage_slot": damage_slot,
            "damage_slot_active": damage_slot in active_slots,
            "primary_damage_slot": primary_damage_slot,
            "primary_damage_slot_active": (
                primary_damage_slot in initial_occupant_slots
            ),
            "damage_np_ready": damage_np_ready,
            "damage_np_percent": damage_np_percent,
            "damage_np_source": damage_np_source,
            "damage_np_needs_charge": damage_np_needs_charge,
            "charge_skill_threshold": charge_threshold,
            "enemy_count": len(alive_enemies),
            "max_enemy_charge": max_enemy_charge,
            "after_aoe_np": self._aoe_np_fired,
            "support_id": (
                str(self.selected_support.candidate.get("id"))
                if (
                    self.selected_support is not None
                    and int(
                        self.selected_support.candidate.get(
                            "party_slot",
                            self.config.get("battle", {}).get(
                                "support_slot",
                                3,
                            ),
                        )
                    )
                    in initial_occupant_slots
                )
                else None
            ),
        }

    def _skill_group_matches(
        self,
        group: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        when = group.get("when", {})
        if (
            "turn_at_least" in when
            and context["turn"] < int(when["turn_at_least"])
        ):
            return False
        if (
            "turn_at_most" in when
            and context["turn"] > int(when["turn_at_most"])
        ):
            return False
        if (
            "damage_np_ready" in when
            and bool(context["damage_np_ready"])
            != bool(when["damage_np_ready"])
        ):
            return False
        if (
            "damage_slot_active" in when
            and bool(context["damage_slot_active"])
            != bool(when["damage_slot_active"])
        ):
            return False
        if (
            "primary_damage_slot_active" in when
            and bool(context["primary_damage_slot_active"])
            != bool(when["primary_damage_slot_active"])
        ):
            return False
        if (
            "enemy_count_at_least" in when
            and context["enemy_count"] < int(when["enemy_count_at_least"])
        ):
            return False
        if (
            "enemy_count_at_most" in when
            and context["enemy_count"] > int(when["enemy_count_at_most"])
        ):
            return False
        if (
            "active_party_count_at_least" in when
            and context["active_party_count"]
            < int(when["active_party_count_at_least"])
        ):
            return False
        if (
            "active_party_count_at_most" in when
            and context["active_party_count"]
            > int(when["active_party_count_at_most"])
        ):
            return False
        if (
            "enemy_charge_at_least" in when
            and context["max_enemy_charge"]
            < float(when["enemy_charge_at_least"])
        ):
            return False
        if (
            "after_aoe_np" in when
            and context["after_aoe_np"] != bool(when["after_aoe_np"])
        ):
            return False
        if (
            "damage_slot_equals" in when
            and context["damage_slot"] != int(when["damage_slot_equals"])
        ):
            return False
        if (
            "active_slot" in when
            and int(when["active_slot"]) not in context["active_slots"]
        ):
            return False
        current_active_slots = list(context.get("active_slots", []))
        initial_occupant_slots = set(
            context.get(
                "initial_occupant_slots",
                current_active_slots
                if context.get("primary_damage_slot_active", True)
                else [],
            )
        )
        replaced_slots = set(
            context.get(
                "replaced_slots",
                [
                    slot
                    for slot in current_active_slots
                    if slot not in initial_occupant_slots
                ],
            )
        )
        if (
            "initial_occupant_slot" in when
            and int(when["initial_occupant_slot"])
            not in initial_occupant_slots
        ):
            return False
        if (
            "replaced_slot" in when
            and int(when["replaced_slot"]) not in replaced_slots
        ):
            return False
        if "owner_hp_at_most" in when:
            owner_slot = int(
                group.get(
                    "skill_owner_slot",
                    when.get("active_slot", context["damage_slot"]),
                )
            )
            owner_hp = context.get("party_hp_ratios", {}).get(
                owner_slot,
                1.0,
            )
            if owner_hp > float(when["owner_hp_at_most"]):
                return False
        if (
            "any_party_hp_at_most" in when
            and context.get("min_party_hp_ratio", 1.0)
            > float(when["any_party_hp_at_most"])
        ):
            return False
        if (
            "support_ids" in when
            and context["support_id"]
            not in {str(value) for value in when["support_ids"]}
        ):
            return False
        if str(group.get("rule_key", "")) == "use_charge_skills":
            owner_slot = group.get("charge_owner_slot")
            if owner_slot is not None:
                owner_slot = int(owner_slot)
                # Once replacements are on the field, the same button
                # coordinate can represent a different skill. Do not carry
                # the original fixed-team charge-only gate onto substitutes.
                if context.get("primary_damage_slot_active", True):
                    threshold = int(
                        self.config.get("battle", {}).get(
                            "all_servant_charge_skill_threshold_percent",
                            100,
                        )
                    )
                    owner_np = context.get("np_percent_by_slot", {}).get(
                        owner_slot
                    )
                    if owner_np is not None and owner_np >= threshold:
                        return False
                    if (
                        owner_np is None
                        and owner_slot == context.get("damage_slot")
                        and not context.get("damage_np_needs_charge", True)
                    ):
                        return False
            elif not context.get("damage_np_needs_charge", True):
                return False
        if str(group.get("rule_key", "")) in {
            "use_evasion_skills",
            "use_guts_skills",
        } and bool(group.get("require_survival_need", False)):
            owner_slot = int(
                group.get("skill_owner_slot", context["damage_slot"])
            )
            owner_hp = context.get("party_hp_ratios", {}).get(
                owner_slot,
                1.0,
            )
            hp_threshold = float(
                self.config.get("battle", {}).get(
                    "survival_skill_hp_threshold",
                    0.65,
                )
            )
            charge_threshold = float(
                self.config.get("battle", {}).get(
                    "survival_enemy_charge_threshold",
                    0.35,
                )
            )
            survival_needed = (
                owner_hp <= hp_threshold
                or context.get("min_party_hp_ratio", 1.0) <= hp_threshold
                or context.get("max_enemy_charge", 0.0)
                >= charge_threshold
                or context.get("active_party_count", 3) <= 2
            )
            if not survival_needed:
                return False
        return True

    def _skill_group_ready(
        self,
        group: dict[str, Any],
        screen: np.ndarray,
    ) -> bool:
        points = group.get("requires_skill_points")
        if points is None:
            points = [
                step["tap"]
                for step in group.get("steps", [])
                if "tap" in step and step.get("check_cooldown", True)
            ]
        return all(
            self.battle_planner.skill_ready(screen, point)
            for point in points
        )

    def _configured_skill_groups(self) -> list[dict[str, Any]]:
        if (
            not self._battle_rule_enabled("use_skills")
            or not self._battle_rule_enabled("use_all_frontline_skills")
        ):
            return []
        groups: list[dict[str, Any]] = []
        battle = self.config.get("battle", {})
        frontline_points = battle.get(
            "frontline_skill_points",
            battle.get("fallback_damage_skill_points", {}),
        )
        for raw_slot, points in frontline_points.items():
            slot = int(raw_slot)
            for index, point in enumerate(points, start=1):
                groups.append(
                    {
                        "id": f"frontline_slot{slot}_skill{index}",
                        "_runtime_id": (
                            f"frontline:slot{slot}:skill{index}"
                        ),
                        "name": f"前排位置{slot}技能{index}",
                        "reason": (
                            "当前人物技能按钮已冷却；"
                            "不推断人物身份或技能类型"
                        ),
                        "rule_key": "use_all_frontline_skills",
                        "skill_owner_slot": slot,
                        "repeat_after_cooldown": True,
                        "auto_target_damage_dealer": True,
                        "when": {"active_slot": slot},
                        "requires_skill_points": [point],
                        "steps": [
                            {
                                "tap": point,
                                "after": 2.0,
                            }
                        ],
                    }
                )
        return groups

    @staticmethod
    def _skill_group_points(
        group: dict[str, Any],
    ) -> set[tuple[int, int]]:
        points = group.get("requires_skill_points")
        if points is None:
            points = [
                step["tap"]
                for step in group.get("steps", [])
                if "tap" in step and step.get("check_cooldown", True)
            ]
        return {
            (round(float(point[0])), round(float(point[1])))
            for point in points
        }

    def _skill_group_priority(
        self,
        group: dict[str, Any],
        context: dict[str, Any],
    ) -> int:
        rule_key = str(group.get("rule_key", ""))
        if rule_key in {"use_evasion_skills", "use_guts_skills"}:
            return 0
        if (
            context.get(
                "damage_np_needs_charge",
                not context["damage_np_ready"],
            )
            and rule_key == "use_charge_skills"
            and self._battle_rule_enabled("prioritize_charge_for_main")
        ):
            return 1
        if rule_key == "use_attack_buffs":
            return 2
        if rule_key == "use_charge_skills":
            return 3
        if rule_key == "use_support_skills":
            return 4
        if rule_key == "use_all_frontline_skills":
            return 5
        return 4

    def _skill_group_repeats(self, group: dict[str, Any]) -> bool:
        return (
            bool(group.get("repeat_after_cooldown", False))
            and self._battle_rule_enabled("reuse_skills_after_cooldown")
        )

    def _run_conditional_skill_phase(
        self,
        screen: np.ndarray,
        groups: list[dict[str, Any]],
    ) -> tuple[np.ndarray, list[str]]:
        current = screen
        used_this_turn: set[str] = set()
        used_skill_points: set[tuple[int, int]] = set()
        executed: list[str] = []
        max_groups = int(
            self.config.get("battle", {}).get(
                "max_skill_groups_per_turn",
                12,
            )
        )
        initial = self._battle_skill_context(current)
        np_value = initial["damage_np_percent"]
        np_text = (
            f"{np_value}%（{initial['damage_np_source']}）"
            if np_value is not None
            else "未识别"
        )
        self.log(
            "技能判定："
            f"主力位{initial['damage_slot']}宝具值{np_text}，"
            f"{'需要' if initial['damage_np_needs_charge'] else '无需'}"
            f"充能（阈值{initial['charge_skill_threshold']}%），"
            f"敌人{initial['enemy_count']}名，"
            f"最高充能评分{initial['max_enemy_charge']:.3f}，"
            f"前排{initial['active_party_count']}名"
        )

        for _ in range(max_groups):
            context = self._battle_skill_context(current)
            eligible: list[tuple[int, int, dict[str, Any]]] = []
            for order, group in enumerate(groups):
                runtime_id = str(group["_runtime_id"])
                if runtime_id in used_this_turn:
                    continue
                if self._skill_group_points(group) & used_skill_points:
                    continue
                repeat = self._skill_group_repeats(group)
                if not repeat and runtime_id in self._used_skill_groups:
                    continue
                if not self._skill_group_matches(group, context):
                    continue
                if not self._skill_group_ready(group, current):
                    continue
                if not group.get("steps"):
                    continue
                eligible.append(
                    (
                        self._skill_group_priority(group, context),
                        order,
                        group,
                    )
                )
            if not eligible:
                break
            _, _, selected = min(
                eligible,
                key=lambda item: (item[0], item[1]),
            )

            runtime_id = str(selected["_runtime_id"])
            name = str(selected.get("name") or selected.get("id") or runtime_id)
            reason = str(selected.get("reason", "满足本回合技能条件"))
            self.log(f"释放技能组：{name}（{reason}）")
            used_this_turn.add(runtime_id)
            used_skill_points.update(self._skill_group_points(selected))
            if not self._skill_group_repeats(selected):
                self._used_skill_groups.add(runtime_id)
            current = self._perform_steps(
                list(selected.get("steps", [])),
                current,
                f"第{self.battle_turn}回合·{name}",
                confirm_skills=self._battle_rule_enabled(
                    "confirm_skill_use"
                ),
                accelerate_skill_animation=True,
            )
            executed.append(name)
            if self.stop_event.is_set():
                break
            self._wait_interruptibly(
                float(
                    self.config.get("battle", {}).get(
                        "after_skill_group_wait_seconds",
                        0.8,
                    )
                )
            )
            current = self.device.capture()
            current, _ = self._resolve_skill_confirmation(
                current,
                f"第{self.battle_turn}回合·{name}补偿检查",
                accelerate_animation=True,
            )
            skill_error_rule = next(
                (
                    rule
                    for rule in self.config.get("rules", [])
                    if rule.get("name") == "skill_unavailable_close"
                ),
                None,
            )
            skill_error_match = (
                self.matcher.match_rule(current, skill_error_rule)
                if skill_error_rule is not None
                else None
            )
            if skill_error_match is not None:
                self.log(
                    f"{name}当前不可使用，自动点击提示框“关闭”并继续本回合"
                )
                self._tap(skill_error_match.center, "关闭技能不可用提示")
                self._wait_interruptibly(
                    float(
                        self.config.get("battle", {}).get(
                            "skill_error_close_wait_seconds",
                            1.0,
                        )
                    )
                )
                current = self.device.capture()
                current = self._wait_until_battle_ready_after_skill(
                    current,
                    f"第{self.battle_turn}回合·{name}",
                    target_slot=int(context["damage_slot"]),
                    active_slots=list(context["active_slots"]),
                )
                continue
            if bool(selected.get("auto_target_damage_dealer", False)):
                current, resolved_target = self._resolve_pending_skill_target(
                    current,
                    name,
                    target_slot=int(context["damage_slot"]),
                    active_slots=list(context["active_slots"]),
                )
                if resolved_target:
                    current = self._wait_until_battle_ready_after_skill(
                        current,
                        f"第{self.battle_turn}回合·{name}",
                        target_slot=int(context["damage_slot"]),
                        active_slots=list(context["active_slots"]),
                    )
                    continue
                target_rule = next(
                    (
                        rule
                        for rule in self.config.get("rules", [])
                        if rule.get("action") == "target_damage_dealer"
                    ),
                    None,
                )
                if (
                    target_rule is not None
                    and self.matcher.match_rule(current, target_rule)
                    is not None
                ):
                    slot = int(context["damage_slot"])
                    point = self._skill_target_point(
                        current,
                        slot,
                        list(context["active_slots"]),
                    )
                    self._tap(
                        point,
                        f"{name}自动选择当前输出位 {slot}",
                    )
                    self._wait_with_skill_animation_tap(
                        float(
                            self.config.get("battle", {}).get(
                                "after_skill_target_wait_seconds",
                                1.2,
                            )
                        ),
                        current,
                        f"{name}自动选人后",
                        start_delay_seconds=float(
                            self.config.get("battle", {}).get(
                                "skill_animation_after_target_tap_delay_seconds",
                                0.08,
                            )
                        ),
                        animation_started_by_previous_tap=True,
                    )
                    current = self.device.capture()

            current = self._wait_until_battle_ready_after_skill(
                current,
                f"第{self.battle_turn}回合·{name}",
                target_slot=int(context["damage_slot"]),
                active_slots=list(context["active_slots"]),
            )

        if len(executed) >= max_groups:
            self._pause(
                f"单回合技能组达到安全上限 {max_groups}",
                current,
            )
        if executed:
            self.log(
                f"本回合技能判定完成：{'、'.join(executed)}；准备点击攻击"
            )
        else:
            self.log("本回合无需释放技能或所需技能仍在冷却；准备点击攻击")
        return current, executed

    def _verified_np_slots(
        self,
        pre_attack_screen: np.ndarray,
        card_screen: np.ndarray,
        candidate_slots: list[int],
    ) -> list[int]:
        minimum_percent = int(
            self.config.get("battle", {}).get(
                "np_release_min_percent",
                100,
            )
        )
        verified: list[int] = []
        for slot in candidate_slots:
            if slot in self._blocked_np_slots:
                self.log(
                    f"{slot}号位宝具本回合曾被游戏拒绝，"
                    "已加入临时黑名单并改选普通指令卡"
                )
                continue
            charge_percent, charge_source = (
                self.battle_planner.np_charge_percent(
                    pre_attack_screen,
                    slot,
                )
            )
            disabled_detector = getattr(
                self.battle_planner,
                "np_card_disabled",
                None,
            )
            disabled, disabled_confidence = (
                disabled_detector(card_screen, slot)
                if callable(disabled_detector)
                else (False, 0.0)
            )
            if disabled:
                self.log(
                    f"{slot}号位宝具卡虽然出现，但检测到"
                    "“无法行动/睡眠/ERROR/宝具封印”"
                    f"（{disabled_confidence:.3f}）；硬排除且不点击"
                )
                continue
            card_selectable = self.battle_planner.np_card_selectable(
                card_screen,
                slot,
            )
            if charge_percent is None:
                self.log(
                    f"{slot}号位攻击前宝具值无法可靠读取，"
                    "保守排除宝具候选"
                )
                continue
            if charge_percent < minimum_percent:
                visual_note = (
                    "；指令卡页虽检测到亮区，但按背景误判排除"
                    if card_selectable
                    else ""
                )
                self.log(
                    f"{slot}号位攻击前宝具值 {charge_percent}%"
                    f"（{charge_source}），未满 {minimum_percent}%"
                    f"{visual_note}"
                )
                continue
            if not card_selectable:
                self.log(
                    f"{slot}号位攻击前宝具值 {charge_percent}%"
                    f"（{charge_source}），但指令卡页没有可点击宝具卡；"
                    "按封印或不可用处理"
                )
                continue
            self.log(
                f"{slot}号位宝具双重确认通过：攻击前 "
                f"{charge_percent}%（{charge_source}），"
                "指令卡页可点击"
            )
            verified.append(slot)
        return verified

    def _block_pending_np_slots(
        self,
        reason: str,
        screen: np.ndarray | None = None,
    ) -> None:
        if not self._pending_np_slots:
            return
        blocked = sorted(self._pending_np_slots)
        disabled_detector = getattr(
            self.battle_planner,
            "np_card_disabled",
            None,
        )
        if screen is not None and callable(disabled_detector):
            visibly_disabled = [
                slot
                for slot in blocked
                if disabled_detector(screen, slot)[0]
            ]
            if visibly_disabled:
                blocked = visibly_disabled
        self._blocked_np_slots.update(blocked)
        self._pending_np_slots.clear()
        self.log(
            f"{reason}；本回合禁用宝具槽位 "
            f"{', '.join(map(str, blocked))}，返回后改用可用普通卡"
        )

    def _prioritize_ready_np_slots(
        self,
        ready_np_slots: list[int],
        *,
        np_targets: dict[int, str],
        np_damage: dict[int, bool],
        enemy_count: int,
    ) -> list[int]:
        slots = list(ready_np_slots)
        if not self._battle_rule_enabled("use_support_nps"):
            slots = [slot for slot in slots if np_damage.get(slot, False)]
        if (
            enemy_count >= 2
            and self._battle_rule_enabled("prefer_aoe_for_multiple")
        ):
            slots.sort(
                key=lambda slot: (
                    not (
                        np_damage.get(slot, False)
                        and np_targets.get(slot) == "aoe"
                    ),
                    np_targets.get(slot) == "single",
                )
            )
        elif enemy_count == 1:
            slots.sort(
                key=lambda slot: (
                    not (
                        np_damage.get(slot, False)
                        and np_targets.get(slot) == "single"
                    ),
                    np_targets.get(slot) != "support",
                )
            )
        if (
            len(slots) > 1
            and not self._battle_rule_enabled("use_multiple_ready_nps")
        ):
            return slots[:1]
        if (
            len(slots) > 1
            and self._battle_rule_enabled("prefer_support_np_first")
        ):
            slots.sort(key=lambda slot: np_damage.get(slot, False))
        return slots

    def _priority_enemy_target_if_needed(
        self,
        screen: np.ndarray,
        *,
        attacker_class: str | None,
        single_damage_np_ready: bool,
    ) -> tuple[
        int,
        tuple[int, int],
        float,
        float,
        str | None,
        bool,
    ] | None:
        alive = self.battle_planner.alive_enemy_slots(screen)
        if len(alive) == 1:
            self.log(
                f"仅检测到敌人{alive[0]}仍在场；"
                "游戏会自动锁定唯一目标，禁止点击其他敌人位置"
            )
            return None
        if not alive:
            self.log("指令卡页未确认到存活敌人，不执行敌方目标点击")
            return None

        target = self.battle_planner.priority_enemy(
            screen,
            attacker_class=attacker_class,
            prefer_class_advantage=self._battle_rule_enabled(
                "prefer_class_advantage"
            ),
            prefer_charge=self._battle_rule_enabled(
                "prefer_imminent_charge"
            ),
            prefer_high_hp=(
                single_damage_np_ready
                and self._battle_rule_enabled("single_np_high_hp")
            ),
        )
        if target is not None and target[0] not in alive:
            self.log(
                f"目标复核发现敌人{target[0]}已经退场；"
                "取消本次目标点击"
            )
            return None
        return target

    def _sync_battle_turn(self, screen: np.ndarray) -> None:
        detect_turn = getattr(
            self.battle_planner,
            "battle_turn_number",
            None,
        )
        detected = detect_turn(screen) if callable(detect_turn) else None
        if detected is None:
            self.battle_turn += 1
            self.log(
                "未可靠识别游戏右上角回合数，"
                f"使用内部增量兜底为第 {self.battle_turn} 回合"
            )
            return

        previous = self.battle_turn
        self.battle_turn = int(detected)
        if previous > 0 and previous != self.battle_turn:
            self._blocked_np_slots.clear()
            self._pending_np_slots.clear()
            self.log(
                f"按游戏右上角真实回合数校正："
                f"内部 {previous} → 游戏 {self.battle_turn}"
            )

    def _battle_action(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> None:
        battle = self.config.get("battle", {})
        resume_card_screen = str(match.rule.get("name", "")) == "command_cards"
        if resume_card_screen:
            detect_turn = getattr(
                self.battle_planner,
                "battle_turn_number",
                None,
            )
            detected_turn = (
                detect_turn(screen) if callable(detect_turn) else None
            )
            if detected_turn is not None:
                self.battle_turn = int(detected_turn)
            elif self.battle_turn <= 0:
                self.battle_turn = 1
            self.log(
                "检测到已经打开的指令卡界面，直接继续分析和选卡；"
                "禁止点击右下角“返回”"
            )
        else:
            self._sync_battle_turn(screen)
        max_turns = int(battle.get("max_turns", 15))
        if self.battle_turn > max_turns:
            self._pause(
                f"战斗超过 {max_turns} 回合，按特殊关卡暂停",
                screen,
                match,
            )

        self.log(f"简单战斗：第 {self.battle_turn} 回合")
        if not resume_card_screen:
            initial_context = self._battle_skill_context(screen)
            should_pause_for_missing_primary = (
                bool(
                    battle.get(
                        "pause_if_primary_damage_slot_inactive",
                        False,
                    )
                )
                or not self._battle_rule_enabled(
                    "continue_with_replacement"
                )
            )
            if (
                should_pause_for_missing_primary
                and not initial_context["primary_damage_slot_active"]
            ):
                self._pause(
                    "固定队伍主力输出位已退场，按特殊/高难战斗暂停",
                    screen,
                    match,
                )
        turn_plan = battle.get("turns", {}).get(str(self.battle_turn), {})
        command_screen = screen
        skill_groups = (
            [] if resume_card_screen else self._configured_skill_groups()
        )
        if skill_groups:
            command_screen, _ = self._run_conditional_skill_phase(
                command_screen,
                skill_groups,
            )
        else:
            steps: list[dict[str, Any]] = []
            active_slots = (
                []
                if resume_card_screen
                else self.battle_planner.active_party_slots(screen)
            )
            _, damage_slot = self._damage_dealer_point(screen)
            damage_np_percent, damage_np_source = (
                self.battle_planner.np_charge_percent(screen, damage_slot)
            )
            charge_threshold = int(
                battle.get("charge_skill_threshold_percent", 100)
            )
            damage_np_needs_charge = (
                damage_slot in active_slots
                and (
                    damage_np_percent is None
                    or damage_np_percent < charge_threshold
                )
            )
            should_recharge = (
                not self._post_np_charge_used
                and damage_slot in active_slots
                and damage_np_needs_charge
                and (self._aoe_np_fired or self.battle_turn >= 2)
            )
            if should_recharge:
                steps.extend(battle.get("post_aoe_np_charge_steps", []))
                self._post_np_charge_used = True
                damage_np_text = (
                    f"{damage_np_percent}%"
                    if damage_np_percent is not None
                    else "未识别"
                )
                self.log(
                    f"主力输出位 {damage_slot} 宝具值{damage_np_text}"
                    f"（{damage_np_source}），低于{charge_threshold}%，"
                    "使用保留的群体充能与定向充能技能"
                )
            elif (
                len(active_slots) == 1
                and not self._single_survivor_skills_used
            ):
                steps.extend(battle.get("single_survivor_steps", []))
                self._single_survivor_skills_used = True
                self.log("检测到仅剩一名前排，从配置使用回避/攻击强化技能")
            elif self.battle_turn == 1 and not resume_card_screen:
                steps.extend(battle.get("opening_steps", []))
                if self.selected_support is not None:
                    steps.extend(
                        self.selected_support.candidate.get(
                            "opening_steps",
                            [],
                        )
                    )
                if damage_np_needs_charge:
                    steps.extend(battle.get("opening_charge_steps", []))
                    if self.selected_support is not None:
                        steps.extend(
                            self.selected_support.candidate.get(
                                "opening_charge_steps",
                                [],
                            )
                        )
            self._perform_steps(
                steps,
                screen,
                f"第{self.battle_turn}回合",
                confirm_skills=self._battle_rule_enabled(
                    "confirm_skill_use"
                ),
                accelerate_skill_animation=True,
            )
            if steps and not self.stop_event.is_set():
                self._wait_interruptibly(
                    float(battle.get("after_skills_wait_seconds", 0.9))
                )
                command_screen = self.device.capture()

        pre_attack_steps = (
            []
            if resume_card_screen
            else list(turn_plan.get("pre_attack", []))
        )
        if pre_attack_steps:
            self._perform_steps(
                pre_attack_steps,
                command_screen,
                f"第{self.battle_turn}回合·指定动作",
            )
            self._wait_interruptibly(
                float(battle.get("after_skills_wait_seconds", 0.9))
            )
            command_screen = self.device.capture()

        if (
            not resume_card_screen
            and self._battle_rule_enabled("confirm_skill_use")
        ):
            command_screen, recovered_confirmation = (
                self._resolve_skill_confirmation(
                    command_screen,
                    f"第{self.battle_turn}回合·Attack前",
                    accelerate_animation=True,
                )
            )
            if (
                recovered_confirmation
                and self._skill_target_overlay_visible(command_screen)
            ):
                recovery_context = self._battle_skill_context(command_screen)
                recovery_slot = int(recovery_context["damage_slot"])
                recovery_point = self._skill_target_point(
                    command_screen,
                    recovery_slot,
                    list(recovery_context["active_slots"]),
                )
                self._tap(
                    recovery_point,
                    "Attack前遗留单体技能选择当前输出位",
                )
                self._wait_with_skill_animation_tap(
                    float(battle.get("after_skill_target_wait_seconds", 1.2)),
                    command_screen,
                    "Attack前遗留单体技能选人后",
                    start_delay_seconds=float(
                        battle.get(
                            "skill_animation_after_target_tap_delay_seconds",
                            0.08,
                        )
                    ),
                    animation_started_by_previous_tap=True,
                )
                if not self.stop_event.is_set():
                    command_screen = self.device.capture()

        command_screen, _ = self._resolve_pending_skill_target(
            command_screen,
            "Attack前遗留的单体技能",
        )

        support_slot = int(
            (
                self.selected_support.candidate.get("party_slot")
                if self.selected_support is not None
                else None
            )
            or battle.get("support_slot", 3)
        )
        active_slots = self.battle_planner.active_party_slots(command_screen)
        if not active_slots:
            raise PauseRequested(
                "攻击前未检测到任何在场从者，已暂停以避免操作空位置"
            )
        primary_damage_slot = int(battle.get("primary_damage_slot", 3))
        initial_occupant_slots = {
            slot for slot in active_slots if self._party_slot_is_initial(slot)
        }
        replaced_slots = set(active_slots) - initial_occupant_slots
        support_active = support_slot in initial_occupant_slots
        primary_damage_active = (
            primary_damage_slot in initial_occupant_slots
        )
        _, fallback_damage_slot = self._damage_dealer_point(command_screen)
        if not primary_damage_active:
            self.log(
                f"固定主力位 {primary_damage_slot} 已退场，"
                f"改用替补输出位 {fallback_damage_slot}"
            )
        configured_np_slots = {
            int(slot) for slot in battle.get("np_use_slots", [support_slot])
        }
        np_candidate_slots = [
            slot
            for slot in active_slots
            if slot in configured_np_slots
        ]
        if not self._battle_rule_enabled("use_ready_np"):
            np_candidate_slots = []
        if support_slot in np_candidate_slots:
            np_candidate_slots.remove(support_slot)
            np_candidate_slots.insert(0, support_slot)

        np_profiles = battle.get("np_profiles", {})
        np_targets: dict[int, str] = {}
        np_colors: dict[int, str | None] = {}
        np_damage: dict[int, bool] = {}
        for slot in np_candidate_slots:
            profile = np_profiles.get(str(slot), {})
            target = str(profile.get("target", "support")).lower()
            color = str(profile.get("color", "")).lower() or None
            damage = bool(
                profile.get("damage", target in {"single", "aoe"})
            )
            if (
                self.selected_support is not None
                and slot == support_slot
                and support_active
            ):
                target = str(
                    self.selected_support.candidate.get("np_target", target)
                ).lower()
                color = (
                    str(
                        self.selected_support.candidate.get(
                            "np_color",
                            color or "",
                        )
                    ).lower()
                    or None
                )
                damage = bool(
                    self.selected_support.candidate.get(
                        "np_damage",
                        target in {"single", "aoe"},
                    )
                )
            if slot in replaced_slots:
                fallback_profile = battle.get("fallback_np_profile", {})
                target = str(
                    fallback_profile.get("target", "single")
                ).lower()
                color = (
                    str(fallback_profile.get("color", "")).lower()
                    or None
                )
                damage = bool(fallback_profile.get("damage", True))
            np_targets[slot] = target
            np_colors[slot] = color
            np_damage[slot] = damage

        alive_enemy_slots = self.battle_planner.alive_enemy_slots(
            command_screen
        )
        enemy_count = len(alive_enemy_slots)

        card_screen: np.ndarray | None = None
        cards_info = None
        attempts = int(battle.get("card_screen_attempts", 5))
        attack_attempts = int(battle.get("attack_tap_attempts", 3))
        command_rule = next(
            (
                rule
                for rule in self.config.get("rules", [])
                if rule.get("name") == "command_cards"
            ),
            None,
        )
        if resume_card_screen:
            for attempt in range(1, attempts + 1):
                if self.stop_event.is_set():
                    return
                card_screen = (
                    screen if attempt == 1 else self.device.capture()
                )
                if (
                    command_rule is not None
                    and self.matcher.match_rule(card_screen, command_rule) is None
                ):
                    self.log(
                        f"等待指令卡页完整出现 {attempt}/{attempts}"
                    )
                    self._wait_interruptibly(
                        float(
                            battle.get("card_screen_retry_seconds", 0.8)
                        )
                    )
                    continue
                try:
                    cards_info = self.battle_planner.analyze_cards(card_screen)
                    break
                except BattleVisionError as exc:
                    self.log(
                        f"等待指令卡界面 {attempt}/{attempts}：{exc}"
                    )
                    self._wait_interruptibly(
                        float(
                            battle.get("card_screen_retry_seconds", 0.8)
                        )
                    )
        else:
            for attack_attempt in range(1, attack_attempts + 1):
                self._tap(
                    match.center,
                    f"Attack {attack_attempt}/{attack_attempts}",
                )
                self._wait_interruptibly(
                    float(battle.get("card_wait_seconds", 1.4))
                )
                if self.stop_event.is_set():
                    return
                for attempt in range(1, attempts + 1):
                    if self.stop_event.is_set():
                        return
                    card_screen = self.device.capture()
                    if (
                        command_rule is not None
                        and self.matcher.match_rule(card_screen, command_rule)
                        is None
                    ):
                        self.log(
                            f"等待指令卡页完整出现 {attempt}/{attempts}"
                        )
                        self._wait_interruptibly(
                            float(
                                battle.get(
                                    "card_screen_retry_seconds",
                                    0.8,
                                )
                            )
                        )
                        continue
                    try:
                        cards_info = self.battle_planner.analyze_cards(
                            card_screen
                        )
                        break
                    except BattleVisionError as exc:
                        self.log(
                            f"等待指令卡界面 {attempt}/{attempts}：{exc}"
                        )
                        self._wait_interruptibly(
                            float(
                                battle.get(
                                    "card_screen_retry_seconds",
                                    0.8,
                                )
                            )
                        )
                if cards_info is not None:
                    break
                if (
                    card_screen is None
                    or self.matcher.match_rule(card_screen, match.rule) is None
                ):
                    break
                self.log(
                    f"Attack 未生效，准备重试 "
                    f"{attack_attempt}/{attack_attempts}"
                )
                self._wait_interruptibly(
                    float(battle.get("attack_retry_seconds", 1.0))
                )
        if cards_info is None or card_screen is None:
            pause_screen = card_screen if card_screen is not None else screen
            self._pause("无法可靠识别五张指令卡", pause_screen)

        available_np_slots = self._verified_np_slots(
            command_screen,
            card_screen,
            np_candidate_slots,
        )
        unavailable_np_slots = [
            slot
            for slot in np_candidate_slots
            if slot not in available_np_slots
        ]
        excluded_support_nps = [
            slot
            for slot in available_np_slots
            if (
                not self._battle_rule_enabled("use_support_nps")
                and not np_damage.get(slot, False)
            )
        ]
        command_np_candidates = [
            slot
            for slot in available_np_slots
            if slot not in excluded_support_nps
        ]
        ready_np_slots = self._prioritize_ready_np_slots(
            command_np_candidates,
            np_targets=np_targets,
            np_damage=np_damage,
            enemy_count=enemy_count,
        )
        if unavailable_np_slots:
            self.log(
                "以下槽位未通过“攻击前宝具值满100% + "
                "指令卡页可点击”双重确认，将改选普通指令卡："
                f"{', '.join(map(str, unavailable_np_slots))}"
            )
        if excluded_support_nps:
            self.log(
                "辅助/弱化型宝具卡可用，但对应规则已关闭："
                f"{', '.join(map(str, excluded_support_nps))}"
            )
        self.log(
            "攻击卡界面可释放宝具槽位："
            f"{', '.join(map(str, ready_np_slots)) or '无'}"
        )
        if (
            enemy_count >= 2
            and any(
                np_damage.get(slot, False)
                and np_targets.get(slot) == "aoe"
                for slot in ready_np_slots
            )
        ):
            self.log(
                f"检测到 {enemy_count} 名在场敌人，群体攻击宝具优先"
            )
        if (
            len(ready_np_slots) > 1
            and not np_damage.get(ready_np_slots[0], False)
        ):
            self.log("辅助/弱化型宝具排在输出宝具之前释放")

        single_damage_np_ready = any(
            np_damage.get(slot, False)
            and np_targets.get(slot) == "single"
            for slot in ready_np_slots
        )
        target_attacker_slot = next(
            (
                slot
                for slot in ready_np_slots
                if np_damage.get(slot, False)
                and np_targets.get(slot) == "single"
            ),
            fallback_damage_slot,
        )
        attacker_class = self._damage_dealer_class(
            card_screen,
            target_attacker_slot,
        )
        current_alive_enemies = self.battle_planner.alive_enemy_slots(
            card_screen
        )
        priority_target = self._priority_enemy_target_if_needed(
            card_screen,
            attacker_class=attacker_class,
            single_damage_np_ready=single_damage_np_ready,
        )
        if priority_target is None:
            if single_damage_np_ready and not current_alive_enemies:
                self._pause(
                    "单体宝具卡可用，但无法确定优先攻击目标",
                    card_screen,
                    match,
                )
        else:
            (
                enemy_slot,
                target_point,
                charge_score,
                health_score,
                enemy_class,
                has_advantage,
            ) = priority_target
            class_text = (
                f"，敌方职阶 {enemy_class}"
                f"{'（克制）' if has_advantage else ''}"
                if enemy_class
                else ""
            )
            self.log(
                f"优先目标：敌人{enemy_slot}（充能评分 "
                f"{charge_score:.3f}，血量条评分 {health_score:.3f}"
                f"{class_text}）"
            )
            self._tap(target_point, "职阶/充能/血量优先目标")
            self._wait_interruptibly(
                float(battle.get("target_wait_seconds", 0.5))
            )
        prefer_primary_output = self._battle_rule_enabled(
            "prefer_primary_output_cards"
        )
        prefer_np_charge_cards = self._battle_rule_enabled(
            "prefer_np_charge_cards"
        )
        prefer_damage_role = self._battle_rule_enabled(
            "prefer_damage_role_over_class"
        )
        primary_output_is_support = (
            fallback_damage_slot == support_slot
            and support_active
            and bool(battle.get("primary_output_is_support", True))
        )
        primary_output_np_percent, primary_output_np_source = (
            self.battle_planner.np_charge_percent(
                command_screen,
                fallback_damage_slot,
            )
        )
        primary_output_needs_charge = (
            prefer_np_charge_cards
            and primary_output_is_support
            and (
                primary_output_np_percent is None
                or primary_output_np_percent < 100
            )
        )
        if primary_output_needs_charge:
            percent_text = (
                f"{primary_output_np_percent}%"
                if primary_output_np_percent is not None
                else "读数未知"
            )
            self.log(
                f"主力输出位置{fallback_damage_slot}宝具值{percent_text}"
                f"（{primary_output_np_source}），优先其Arts/Quick卡加速充能"
            )
        output_cards = [
            card.name
            for card in cards_info
            if primary_output_is_support and card.is_support
        ]
        disabled_cards = [
            card.name for card in cards_info if card.is_disabled
        ]
        if disabled_cards:
            self.log(
                "检测到无法行动指令卡："
                f"{', '.join(disabled_cards)}；"
                "优先选择未禁用角色，数量不足时用禁用卡补齐"
            )
        color_summary = "，".join(
            f"{card.name}:{card.color}/{card.affinity}"
            f"/crit={card.critical_percent if card.critical_percent is not None else '?'}%"
            f"/{'禁用' if card.is_disabled else '可行动'}"
            f"({card.color_confidence:.2f}/{card.affinity_confidence:.2f})"
            for card in cards_info
        )
        self.log(f"卡色/职阶相性识别：{color_summary}")
        if prefer_primary_output and primary_output_is_support:
            self.log(
                "主力输出指令卡识别："
                f"{', '.join(output_cards) if output_cards else '本回合没有主力卡'}"
            )

        explicit_cards = turn_plan.get("cards")
        if explicit_cards:
            if len(explicit_cards) != 3:
                self._pause("指定回合必须恰好选择 3 张卡", card_screen)
            labels = list(explicit_cards)
            plan_reason = "使用指定回合卡序"
        elif len(ready_np_slots) == 1:
            slot = ready_np_slots[0]
            plan = self.battle_planner.choose_with_np(
                cards_info,
                np_label=f"np{slot}",
                np_color=np_colors.get(slot),
                prefer_same_owner=self._battle_rule_enabled(
                    "prefer_same_servant_chain"
                ),
                prefer_same_color=self._battle_rule_enabled(
                    "prefer_same_color_chain"
                ),
                prefer_arts_chain=self._battle_rule_enabled(
                    "prefer_arts_chain"
                ),
                prefer_mighty_chain=self._battle_rule_enabled(
                    "prefer_mighty_chain"
                ),
                prefer_class_advantage=self._battle_rule_enabled(
                    "prefer_card_class_advantage"
                ),
                prefer_primary_output=prefer_primary_output,
                primary_output_is_support=primary_output_is_support,
                prefer_damage_role_over_class=prefer_damage_role,
                prefer_high_critical=self._battle_rule_enabled(
                    "prefer_high_critical_cards"
                ),
                primary_output_needs_charge=primary_output_needs_charge,
            )
            labels = plan.labels
            plan_reason = plan.reason
        elif ready_np_slots:
            plan = self.battle_planner.choose_with_nps(
                cards_info,
                np_labels=[f"np{slot}" for slot in ready_np_slots],
                prefer_same_owner=self._battle_rule_enabled(
                    "prefer_same_servant_chain"
                ),
                prefer_same_color=self._battle_rule_enabled(
                    "prefer_same_color_chain"
                ),
                prefer_arts_chain=self._battle_rule_enabled(
                    "prefer_arts_chain"
                ),
                prefer_mighty_chain=self._battle_rule_enabled(
                    "prefer_mighty_chain"
                ),
                prefer_class_advantage=self._battle_rule_enabled(
                    "prefer_card_class_advantage"
                ),
                prefer_primary_output=prefer_primary_output,
                primary_output_is_support=primary_output_is_support,
                prefer_damage_role_over_class=prefer_damage_role,
                prefer_high_critical=self._battle_rule_enabled(
                    "prefer_high_critical_cards"
                ),
                primary_output_needs_charge=primary_output_needs_charge,
            )
            labels = plan.labels
            plan_reason = plan.reason
        else:
            plan = self.battle_planner.choose_face_cards(
                cards_info,
                charge_priority=prefer_np_charge_cards,
                prefer_same_owner=self._battle_rule_enabled(
                    "prefer_same_servant_chain"
                ),
                prefer_same_color=self._battle_rule_enabled(
                    "prefer_same_color_chain"
                ),
                prefer_arts_chain=self._battle_rule_enabled(
                    "prefer_arts_chain"
                ),
                prefer_mighty_chain=self._battle_rule_enabled(
                    "prefer_mighty_chain"
                ),
                prefer_class_advantage=self._battle_rule_enabled(
                    "prefer_card_class_advantage"
                ),
                prefer_primary_output=prefer_primary_output,
                primary_output_is_support=primary_output_is_support,
                prefer_damage_role_over_class=prefer_damage_role,
                prefer_high_critical=self._battle_rule_enabled(
                    "prefer_high_critical_cards"
                ),
                primary_output_needs_charge=primary_output_needs_charge,
            )
            labels = plan.labels
            plan_reason = plan.reason

        self.log(f"选卡策略：{plan_reason}；顺序 {' → '.join(labels)}")
        # The command-card entrance animation can expose a bright NP card a
        # fraction of a second before the game paints the "cannot act" text.
        # Never trust that transitional frame for the actual tap. Re-capture
        # immediately before selecting any planned NP and fall back to three
        # face cards if even one NP no longer passes full verification.
        planned_np_slots = [
            int(label[2:])
            for label in labels
            if label.startswith("np") and label[2:].isdigit()
        ]
        if planned_np_slots:
            self._wait_interruptibly(
                float(battle.get("np_before_tap_recheck_seconds", 0.5))
            )
            if self.stop_event.is_set():
                return
            final_card_screen = self.device.capture()
            final_np_slots = self._verified_np_slots(
                command_screen,
                final_card_screen,
                planned_np_slots,
            )
            if final_np_slots != planned_np_slots:
                rejected_np_slots = [
                    slot
                    for slot in planned_np_slots
                    if slot not in final_np_slots
                ]
                self._blocked_np_slots.update(rejected_np_slots)
                self.log(
                    "宝具点击前复核发现无法行动/睡眠/ERROR/宝具封印："
                    f"{', '.join(map(str, rejected_np_slots))}；"
                    "取消全部宝具点击并改选三张普通卡"
                )
                cards_info = self.battle_planner.analyze_cards(
                    final_card_screen
                )
                plan = self.battle_planner.choose_face_cards(
                    cards_info,
                    charge_priority=prefer_np_charge_cards,
                    prefer_same_owner=self._battle_rule_enabled(
                        "prefer_same_servant_chain"
                    ),
                    prefer_same_color=self._battle_rule_enabled(
                        "prefer_same_color_chain"
                    ),
                    prefer_arts_chain=self._battle_rule_enabled(
                        "prefer_arts_chain"
                    ),
                    prefer_mighty_chain=self._battle_rule_enabled(
                        "prefer_mighty_chain"
                    ),
                    prefer_class_advantage=self._battle_rule_enabled(
                        "prefer_card_class_advantage"
                    ),
                    prefer_primary_output=prefer_primary_output,
                    primary_output_is_support=primary_output_is_support,
                    prefer_damage_role_over_class=prefer_damage_role,
                    prefer_high_critical=self._battle_rule_enabled(
                        "prefer_high_critical_cards"
                    ),
                    primary_output_needs_charge=primary_output_needs_charge,
                )
                labels = plan.labels
                plan_reason = (
                    "宝具点击前复核发现无法行动/睡眠/ERROR/宝具封印，"
                    "改选三张普通卡；"
                    f"{plan.reason}"
                )
                card_screen = final_card_screen
                ready_np_slots = []

        self._pending_np_slots = {
            int(label[2:])
            for label in labels
            if label.startswith("np") and label[2:].isdigit()
        }
        points = battle.get("card_points", {})
        for label in labels:
            if label not in points:
                self._pause(f"战斗配置没有卡位 {label}", card_screen)
            point = self._scaled_point(points[label], card_screen)
            self._tap(point, f"指令卡 {label}")
            self._wait_interruptibly(float(battle.get("between_cards_seconds", 0.35)))
        if any(np_targets.get(slot) == "aoe" for slot in ready_np_slots):
            self._aoe_np_fired = True

    def _quest_start_action(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> None:
        self.enemy_class = None
        if self._select_highest_level_enabled():
            self.log("等级最高模式：跳过职阶克制分析，直接进入助战页")
        elif self._support_rule_enabled("prefer_class_advantage"):
            detected = self.support_selector.detect_enemy_class(screen)
            if detected is not None:
                self.enemy_class, score = detected
                counters = self.support_selector.counter_classes(
                    self.enemy_class
                )
                if counters:
                    self.log(
                        f"主要敌人职阶：{self.enemy_class}（{score:.3f}）；"
                        f"优先助战职阶：{' / '.join(counters)}"
                    )
                else:
                    self.log(
                        f"识别到特殊职阶 {self.enemy_class}，"
                        "没有统一克制关系，准备使用已启用的兜底策略"
                    )
            elif self._support_rule_enabled("use_berserker_fallback"):
                self.log(
                    "本关主要敌人职阶未识别，按助战策略改选狂阶兜底"
                )
            else:
                self._pause(
                    "无法识别本关主要敌人职阶，且未启用狂阶兜底",
                    screen,
                    match,
                )
        else:
            self.log("助战职阶克制筛选已关闭，使用已配置的兜底职阶")
        self.selected_support = None
        self.support_refreshes = 0
        self._reset_support_class_search()
        self._tap(match.center, "开始任务")

    def _select_forced_support_action(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> None:
        if not self._select_highest_level_enabled():
            self.log(
                "当前页发现“客将从者”；客将数量不能证明没有普通助战，"
                "先完整扫描当前首选列表；翻到最后仍未发现普通助战时，"
                "立即选择客将，不再切换其他职阶列表"
            )
        self._select_support_action(screen, match)
        return

    def _select_guest_only_berserker_page_if_confirmed(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> bool:
        """Select a visually proven guest-only Berserker page first."""
        settings = self.config.get("support", {})
        fallback_class = str(
            settings.get("unknown_enemy_class", "berserker")
        )
        selected_method = getattr(
            self.support_selector,
            "selected_support_filter",
            None,
        )
        selected_filter = (
            selected_method(screen) if callable(selected_method) else None
        )
        if selected_filter is None:
            selected_filter = self.support_filter_class
        if selected_filter != fallback_class:
            return False

        guest_rows_method = getattr(
            self.support_selector,
            "guest_support_row_ranges",
            None,
        )
        if not callable(guest_rows_method):
            return False
        try:
            guest_rows = guest_rows_method(
                screen,
                respect_exclusion=False,
            )
        except TypeError:
            guest_rows = guest_rows_method(screen)

        normal_row_method = getattr(
            self.support_selector,
            "has_normal_support_row",
            None,
        )
        page_has_normal = (
            bool(normal_row_method(screen))
            if callable(normal_row_method)
            else False
        )
        self.support_berserker_guest_seen |= bool(guest_rows)
        self.support_berserker_normal_seen |= page_has_normal
        any_normal_seen = (
            self.support_default_normal_seen
            or self.support_berserker_normal_seen
        )

        forced_only_method = getattr(
            self.support_selector,
            "is_forced_only_screen",
            None,
        )
        visually_guest_only = (
            bool(guest_rows)
            and not page_has_normal
            and callable(forced_only_method)
            and bool(forced_only_method(screen))
        )
        if not visually_guest_only or any_normal_seen:
            return False

        self.support_berserker_scan_complete = True
        self.log(
            "助战前置判断：当前狂阶页只有客将、没有普通助战，"
            "且客将下方为空，单行页同时就是列表末页；"
            "立即选择客将，不再执行等级/宝具条件搜索"
        )
        self._select_guest_after_full_scan(
            screen,
            match,
            guest_rows,
            scan_label="狂阶仅客将前置判定",
        )
        return True

    def _select_guest_after_full_scan(
        self,
        screen: np.ndarray,
        match: MatchResult,
        guest_rows: list[tuple[int, int]],
        *,
        scan_label: str,
    ) -> None:
        if not self._support_rule_enabled("use_forced_support"):
            self._pause(
                f"{scan_label}已完整检查且只有客将，但已关闭"
                "“自动选择关卡限定助战”",
                screen,
                match,
            )
        if not guest_rows:
            self._pause(
                "完整扫描记录为只有客将，但最后一页无法定位客将行",
                screen,
                match,
            )
        settings = self.config.get("support", {})
        candidate = dict(settings.get("forced_support_profile", {}))
        candidate.setdefault("id", "quest_forced_support")
        candidate.setdefault("name", "关卡限定助战")
        candidate.setdefault("party_slot", 3)
        candidate.setdefault("np_target", "aoe")
        candidate.setdefault("np_color", "")
        candidate.setdefault("np_damage", True)
        visual = VisualMatch(
            score=match.score,
            x=match.x,
            y=match.y,
            width=match.width,
            height=match.height,
        )
        self.selected_support = SupportChoice(
            candidate=candidate,
            match=visual,
            enemy_class=self.enemy_class or "forced",
            counter_class="forced",
        )
        first_top, first_bottom = min(guest_rows)
        click_x = self._scaled_point(
            [
                settings.get("forced_support_row_click_x", 750),
                0,
            ],
            screen,
        )[0]
        point = (
            click_x,
            (int(first_top) + int(first_bottom)) // 2,
        )
        self.support_refreshes = 0
        self._reset_support_class_search()
        self.log(
            f"{scan_label}已翻到最后一页；整个扫描过程没有发现普通助战；"
            "选择当前页最上方的可用客将"
        )
        self._tap(point, "关卡默认助战")

    def _build_support_search_stages(
        self,
        default_filter_class: str,
        fallback_class: str,
    ) -> None:
        options = self.config.get("support", {}).get("strategy_options", {})
        use_berserker = bool(options.get("use_berserker_fallback", False))
        use_current_default_list = bool(
            options.get("use_system_recommended_list", False)
        )
        definitions = [
            (
                "search_recommended_level_120",
                "system_default",
                (120, 120),
            ),
            (
                "search_berserker_level_120",
                fallback_class,
                (120, 120),
            ),
            (
                "search_recommended_level_110",
                "system_default",
                (110, 119),
            ),
            (
                "search_berserker_level_110",
                fallback_class,
                (110, 119),
            ),
            (
                "search_recommended_level_100",
                "system_default",
                (100, 109),
            ),
            (
                "search_berserker_level_100",
                fallback_class,
                (100, 109),
            ),
        ]
        has_explicit_search_stages = any(
            key.startswith("search_") for key in options
        )
        if not has_explicit_search_stages:
            # Existing custom profiles predate the six visible stage toggles.
            # Keep their old 120-only scope until those toggles are saved.
            definitions = definitions[:2]

        stages: list[tuple[str, str, tuple[int, int]]] = []
        for rule_key, target, level_range in definitions:
            if not bool(options.get(rule_key, True)):
                continue
            if target == "system_default":
                if not use_current_default_list:
                    continue
            elif not use_berserker:
                continue
            if (
                target == fallback_class
                and default_filter_class == fallback_class
                and use_current_default_list
            ):
                # The default list is already Berserker; scanning the same
                # list twice at one level only wastes time and can loop.
                continue
            stages.append((rule_key, target, level_range))

        self.support_search_rule_keys = [item[0] for item in stages]
        self.support_class_queue = [item[1] for item in stages]
        self.support_search_levels = [item[2] for item in stages]
        self.support_class_index = 0
        self.support_candidate_pass = self._initial_support_candidate_pass(options)
        if stages:
            minimum, maximum = stages[0][2]
            self.support_search_phase = (
                f"level_{minimum}"
                if minimum == maximum
                else f"level_{minimum}_{maximum}"
            )

    @staticmethod
    def _initial_support_candidate_pass(options: dict[str, Any]) -> str:
        prefer_aoe = bool(options.get("prefer_aoe_np", True))
        require_aoe = bool(options.get("require_aoe_np", False))
        return "preferred_aoe" if prefer_aoe or require_aoe else "any_np5"

    def _current_support_level_range(self) -> tuple[int, int]:
        if self.support_search_levels:
            return self.support_search_levels[self.support_class_index]
        # Compatibility for tests or resumed runners created by older builds.
        if self.support_search_phase in {"strict", "generic_120_np5"}:
            return 120, 120
        if self.support_search_phase == "level_110_119":
            return 110, 119
        return 100, 109

    @staticmethod
    def _support_level_requirement(
        minimum_level: int,
        maximum_level: int,
        *,
        require_np5: bool,
        require_aoe: bool,
    ) -> str:
        level_text = (
            f"{minimum_level}级"
            if minimum_level == maximum_level
            else f"{minimum_level}～{maximum_level}级"
        )
        properties = [level_text]
        if require_np5:
            properties.append("宝具5")
        if require_aoe:
            properties.append("全体宝具")
        return "、".join(properties)

    def _select_support_action(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> None:
        settings = self.config.get("support", {})
        options = settings.get("strategy_options", {})
        if self._select_guest_only_berserker_page_if_confirmed(
            screen,
            match,
        ):
            return
        if self._select_highest_level_enabled():
            self._select_highest_level_support(screen, match)
            return
        filter_points = settings.get("class_filter_points", {})
        fallback_class = str(
            settings.get("unknown_enemy_class", "berserker")
        )
        if not self.support_class_queue:
            use_current_default_list = bool(
                options.get("use_system_recommended_list", False)
            )
            if self.enemy_class is None and use_current_default_list:
                detect_enemy = getattr(
                    self.support_selector,
                    "detect_enemy_class",
                    None,
                )
                detected = detect_enemy(screen) if callable(detect_enemy) else None
                if detected is not None:
                    self.enemy_class, score = detected
                    self.log(
                        f"在助战页识别主要敌人职阶："
                        f"{self.enemy_class}（{score:.3f}）"
                    )

            default_filter_class = self.support_default_filter_class
            if not use_current_default_list:
                # “使用系统默认列表”关闭时不再寻找不稳定的“推荐”
                # 图标，直接构建狂阶等级档搜索队列。
                default_filter_class = fallback_class
            else:
                selected_method = getattr(
                    self.support_selector,
                    "selected_support_filter",
                    None,
                )
                if default_filter_class is None:
                    default_filter_class = (
                        selected_method(screen)
                        if callable(selected_method)
                        else None
                    )
            if use_current_default_list and default_filter_class is None:
                has_recommendation_method = getattr(
                    self.support_selector,
                    "has_system_class_recommendation",
                    None,
                )
                has_recommendation = (
                    use_current_default_list
                    and callable(has_recommendation_method)
                    and bool(has_recommendation_method(screen))
                )
                if has_recommendation:
                    default_filter_class = "system_recommended"
                else:
                    desired_method = getattr(
                        self.support_selector,
                        "desired_classes",
                        None,
                    )
                    desired_classes = (
                        desired_method(self.enemy_class)
                        if callable(desired_method)
                        else []
                    )
                    default_filter_class = next(
                        (
                            str(value)
                            for value in desired_classes
                            if str(value) in filter_points
                        ),
                        "all" if "all" in filter_points else "system_recommended",
                    )

            self.support_default_filter_class = str(default_filter_class)
            self.support_recommended_classes = ["system_default"]
            self._build_support_search_stages(
                self.support_default_filter_class,
                fallback_class,
            )
            if not self.support_class_queue:
                self._pause(
                    "助战搜索当前没有可执行的等级/列表阶段",
                    screen,
                    match,
                )
            if (
                fallback_class in self.support_class_queue
                and fallback_class not in filter_points
            ):
                self._pause(
                    "已启用狂阶搜索阶段，但没有配置狂阶筛选坐标",
                    screen,
                    match,
                )
            plan_parts = []
            primary_label = "系统默认"
            for target, (minimum, maximum) in zip(
                self.support_class_queue,
                self.support_search_levels,
            ):
                label = primary_label if target == "system_default" else "狂阶"
                level = (
                    str(minimum)
                    if minimum == maximum
                    else f"{minimum}～{maximum}"
                )
                plan_parts.append(f"{label}{level}级")
            self.log("助战搜索顺序：" + " → ".join(plan_parts))
            if bool(options.get("require_aoe_np", False)):
                self.log("宝具筛选：只允许已登记的群体攻击宝具")
            elif bool(options.get("prefer_aoe_np", True)):
                self.log(
                    "宝具筛选：每个列表和等级档先完整搜索已登记群体宝具；"
                    "没有时回到顶部选择同等级宝具5候选"
                )
            else:
                self.log("宝具筛选：不限制宝具类型，直接选择当前等级档候选")

        desired_class = self.support_class_queue[self.support_class_index]
        use_system_recommended = desired_class == "system_default"
        minimum_level, maximum_level = self._current_support_level_range()
        default_filter_class = str(
            self.support_default_filter_class or fallback_class
        )
        use_current_default_list = bool(
            options.get("use_system_recommended_list", False)
        )
        primary_label = "系统默认"

        if use_system_recommended:
            if default_filter_class == "system_recommended":
                filter_point = settings.get(
                    "system_recommended_filter_point",
                    filter_points.get("all"),
                )
            else:
                filter_point = filter_points.get(default_filter_class)
            if default_filter_class == "system_recommended":
                filter_label = "系统默认推荐列表"
            else:
                filter_label = f"系统默认 {default_filter_class} 列表"
        else:
            filter_point = filter_points.get(desired_class)
            filter_label = f"{desired_class} 列表"
        if filter_point is None:
            self._pause(
                f"没有配置助战筛选按钮：{filter_label}",
                screen,
                match,
            )

        if self.support_filter_class != desired_class:
            self.support_filter_class = desired_class
            self.support_scrolls = 0
            self.support_scroll_stalls = 0
            first_default_scan = (
                use_system_recommended
                and use_current_default_list
                and not self.support_initial_page_scanned
                and self.support_refreshes == 0
            )
            if first_default_scan:
                self.support_initial_page_scanned = True
                self.log(
                    f"首次进入助战页，直接扫描游戏已打开的{filter_label}首屏"
                )
            else:
                self.log(
                    f"切换到{filter_label}并回到顶部，搜索"
                    f"{minimum_level}～{maximum_level}级候选"
                )
                self._tap(
                    self._scaled_point(filter_point, screen),
                    f"筛选 {filter_label}",
                )
                self._wait_interruptibly(
                    float(settings.get("filter_settle_seconds", 1.5))
                )
                return

        first_default_index = next(
            (
                index
                for index, target in enumerate(self.support_class_queue)
                if target == "system_default"
            ),
            -1,
        )
        audit_default = (
            desired_class == "system_default"
            and self.support_class_index == first_default_index
        )
        # Every Berserker level band audits the same visible list.  Keeping
        # this enabled beyond the first 120-level pass prevents a one-row
        # guest-only page from falling into the 110/100 search loop.
        audit_berserker = desired_class == fallback_class
        if default_filter_class == fallback_class and audit_default:
            audit_berserker = True

        current_guest_rows: list[tuple[int, int]] = []
        if audit_default or audit_berserker:
            guest_rows_method = getattr(
                self.support_selector,
                "guest_support_row_ranges",
                None,
            )
            if callable(guest_rows_method):
                try:
                    current_guest_rows = guest_rows_method(
                        screen,
                        respect_exclusion=False,
                    )
                except TypeError:
                    current_guest_rows = guest_rows_method(screen)
            normal_row_method = getattr(
                self.support_selector,
                "has_normal_support_row",
                None,
            )
            page_has_normal = (
                bool(normal_row_method(screen))
                if callable(normal_row_method)
                else False
            )
            if audit_default:
                self.support_default_guest_seen |= bool(current_guest_rows)
                self.support_default_normal_seen |= page_has_normal
            if audit_berserker:
                self.support_berserker_guest_seen |= bool(current_guest_rows)
                if page_has_normal and not self.support_berserker_normal_seen:
                    self.log(
                        "狂阶完整扫描中发现普通玩家助战，"
                        "本关不能判定为限定客将关卡"
                    )
                self.support_berserker_normal_seen |= page_has_normal

            forced_only_method = getattr(
                self.support_selector,
                "is_forced_only_screen",
                None,
            )
            visually_forced_only = (
                audit_berserker
                and bool(current_guest_rows)
                and not page_has_normal
                and callable(forced_only_method)
                and bool(forced_only_method(screen))
            )
            any_normal_seen = (
                self.support_default_normal_seen
                or self.support_berserker_normal_seen
            )
            if visually_forced_only and not any_normal_seen:
                self.support_berserker_scan_complete = True
                self.log(
                    "狂阶当前页只有客将，且客将下方列表区域为空；"
                    "该单行页同时就是列表末页，无需执行空白翻页"
                )
                self._select_guest_after_full_scan(
                    screen,
                    match,
                    current_guest_rows,
                    scan_label="狂阶单行末页",
                )
                return

        prefer_registered = self._support_rule_enabled(
            "prefer_registered_candidates"
        )
        require_np5 = self._support_rule_enabled("require_np5")
        require_aoe = bool(options.get("require_aoe_np", False))
        if not self.support_candidate_pass:
            self.support_candidate_pass = self._initial_support_candidate_pass(
                options
            )
        if require_aoe:
            self.support_candidate_pass = "preferred_aoe"
        pass_requires_aoe = self.support_candidate_pass == "preferred_aoe"
        allowed_class: str | None = (
            default_filter_class
            if use_system_recommended
            and default_filter_class
            not in {"all", "system_recommended"}
            else None if use_system_recommended else desired_class
        )
        choice = None
        registered_method = getattr(
            self.support_selector,
            "choose_registered_level_band",
            None,
        )
        if callable(registered_method) and (
            pass_requires_aoe or prefer_registered
        ):
            choice = registered_method(
                screen,
                allowed_class,
                minimum_level,
                maximum_level,
                require_np5=require_np5,
                require_aoe=pass_requires_aoe,
            )
        elif (
            (pass_requires_aoe or prefer_registered)
            and minimum_level == maximum_level == 120
        ):
            # Compatibility with older selector doubles used by integrations.
            if use_system_recommended:
                legacy_method = getattr(
                    self.support_selector,
                    "choose_from_current_filtered_list",
                    None,
                )
                if callable(legacy_method):
                    choice = legacy_method(screen)
            else:
                legacy_method = getattr(
                    self.support_selector,
                    "choose_from_current_filtered_list",
                    None,
                )
                if callable(legacy_method):
                    choice = legacy_method(screen)

        registered_rejections = getattr(
            self.support_selector,
            "last_registered_rejections",
            [],
        )
        if registered_rejections:
            requirement = self._support_level_requirement(
                minimum_level,
                maximum_level,
                require_np5=require_np5,
                require_aoe=pass_requires_aoe,
            )
            self.log(
                "登记助战模板已匹配，但当前行没有同时通过"
                f"{requirement}复核，已排除："
                + ", ".join(registered_rejections)
            )

        if choice is None and not pass_requires_aoe:
            generic_method = getattr(
                self.support_selector,
                "choose_generic_level_band",
                None,
            )
            if callable(generic_method):
                choice = generic_method(
                    screen,
                    allowed_class or default_filter_class,
                    minimum_level,
                    maximum_level,
                    require_np5=require_np5,
                )
            elif minimum_level == maximum_level == 120:
                legacy_generic = getattr(
                    self.support_selector,
                    "choose_generic_level120_np5",
                    None,
                )
                if callable(legacy_generic):
                    choice = legacy_generic(
                        screen,
                        allowed_class or default_filter_class,
                    )
        if choice is not None:
            self.selected_support = choice
            self.support_refreshes = 0
            candidate = choice.candidate
            counter_method = getattr(
                self.support_selector,
                "counter_classes",
                None,
            )
            actual_counters = (
                counter_method(self.enemy_class)
                if self.enemy_class is not None and callable(counter_method)
                else []
            )
            if use_system_recommended:
                target = f"{primary_label}列表"
            elif (
                self.enemy_class is not None
                and choice.counter_class in actual_counters
            ):
                target = f"克制 {self.enemy_class}"
            elif choice.counter_class == str(
                settings.get("unknown_enemy_class", "berserker")
            ):
                target = "狂阶兜底"
            else:
                target = "符合当前助战设置"
            properties: list[str] = [
                (
                    "系统推荐职阶"
                    if choice.counter_class == "system_recommended"
                    else choice.counter_class
                )
            ]
            actual_level = int(candidate.get("level", 0))
            if actual_level > 0:
                properties.append(f"{actual_level}级")
            elif int(candidate.get("level_at_least", 0)) >= 100:
                properties.append(
                    f"{int(candidate.get('level_at_least', 100))}级以上"
                )
            if int(candidate.get("np_level", 0)) == 5:
                properties.append("宝具5")
            np_target = str(candidate.get("np_target", "unknown")).lower()
            if np_target == "aoe":
                properties.append("全体宝具")
            elif np_target == "single":
                properties.append("单体宝具")
            elif np_target == "support":
                properties.append("辅助宝具")
            else:
                properties.append("宝具类型未登记")
            self.log(
                f"选择助战：{candidate.get('name', candidate.get('id'))} · "
                f"{' · '.join(properties)} · {target} · "
                f"匹配 {choice.match.score:.3f}"
            )
            self.support_scrolls = 0
            self.support_scroll_stalls = 0
            self._tap(choice.match.center, "符合策略的助战")
            return

        max_scrolls = int(settings.get("max_scrolls", 6))
        if (
            self._support_rule_enabled("scroll_support_list")
            and self.support_scrolls < max_scrolls
        ):
            requirement = self._support_level_requirement(
                minimum_level,
                maximum_level,
                require_np5=require_np5,
                require_aoe=pass_requires_aoe,
            )
            self._scroll_support_page(
                screen,
                f"当前页没有{requirement}助战",
                max_scrolls,
            )
            return

        scroll_enabled = self._support_rule_enabled("scroll_support_list")
        reached_list_end = not scroll_enabled or self.support_scrolls >= max_scrolls
        completed_full_scan = scroll_enabled and self.support_scrolls >= max_scrolls
        if reached_list_end:
            if audit_default and completed_full_scan:
                self.support_default_scan_complete = True
            if audit_berserker and completed_full_scan:
                self.support_berserker_scan_complete = True

            # 只有完整扫描狂阶列表才能确认限定客将；系统默认列表中的
            # 客将分布不作为结论。任一已扫描列表见过普通助战仍是反证。
            current_audit_guest_seen = (
                self.support_default_guest_seen
                if audit_default
                else self.support_berserker_guest_seen
            )
            current_audit_normal_seen = (
                self.support_default_normal_seen
                if audit_default
                else self.support_berserker_normal_seen
            )
            any_normal_seen = (
                self.support_default_normal_seen
                or self.support_berserker_normal_seen
            )
            if audit_default:
                scan_label = (
                    "狂阶列表"
                    if default_filter_class == fallback_class
                    else f"{primary_label}列表"
                )
            else:
                scan_label = "狂阶列表"
            audit_can_prove_forced = audit_berserker
            if (
                audit_can_prove_forced
                and completed_full_scan
                and current_audit_guest_seen
                and not current_audit_normal_seen
                and not any_normal_seen
                and current_guest_rows
            ):
                self._select_guest_after_full_scan(
                    screen,
                    match,
                    current_guest_rows,
                    scan_label=scan_label,
                )
                return

        if self.support_candidate_pass == "preferred_aoe" and not require_aoe:
            self.support_candidate_pass = "any_np5"
            self.support_scrolls = 0
            self.support_scroll_stalls = 0
            self.support_filter_class = None
            target_label = (
                primary_label if desired_class == "system_default" else "狂阶"
            )
            level_requirement = self._support_level_requirement(
                minimum_level,
                maximum_level,
                require_np5=require_np5,
                require_aoe=False,
            )
            self.log(
                f"{target_label}列表已完整检查且没有已登记的全体宝具候选；"
                f"回到该列表顶部，继续选择{level_requirement}，"
                "允许单体宝具或宝具类型未登记的优质助战"
            )
            return

        if self.support_class_index + 1 < len(self.support_class_queue):
            previous_target = desired_class
            previous_requirement = self._support_level_requirement(
                minimum_level,
                maximum_level,
                require_np5=require_np5,
                require_aoe=pass_requires_aoe,
            )
            self.support_class_index += 1
            next_target = self.support_class_queue[self.support_class_index]
            next_minimum, next_maximum = self._current_support_level_range()
            self.support_candidate_pass = self._initial_support_candidate_pass(
                options
            )
            self.support_search_phase = (
                f"level_{next_minimum}"
                if next_minimum == next_maximum
                else f"level_{next_minimum}_{next_maximum}"
            )
            self.support_scrolls = 0
            self.support_scroll_stalls = 0
            self.support_filter_class = None
            previous_label = (
                primary_label
                if previous_target == "system_default"
                else "狂阶"
            )
            next_label = (
                primary_label if next_target == "system_default" else "狂阶"
            )
            next_requirement = self._support_level_requirement(
                next_minimum,
                next_maximum,
                require_np5=require_np5,
                require_aoe=(self.support_candidate_pass == "preferred_aoe"),
            )
            self.log(
                f"{previous_label}列表已完整检查且没有{previous_requirement}；"
                f"继续检查{next_label}列表的{next_requirement}"
            )
            return

        max_refreshes = int(settings.get("max_refreshes", 3))
        if not self._support_rule_enabled("refresh_support_list"):
            self._pause(
                "当前列表没有符合勾选条件的助战，且已关闭列表刷新",
                screen,
                match,
            )
        if self.support_refreshes >= max_refreshes:
            self._pause(
                f"刷新 {self.support_refreshes} 次仍未找到符合条件的助战",
                screen,
                match,
            )
        point = self._scaled_point(
            settings.get("refresh_point", [1450, 110]),
            screen,
        )
        self.support_refreshes += 1
        # 列表更新会保留原滚动位置。保留最初由游戏打开的默认职阶，
        # 下次重新点该职阶筛选，强制从列表顶部重新执行已启用阶段。
        default_filter_class = self.support_default_filter_class
        self._reset_support_class_search(reset_initial_page=False)
        self.support_default_filter_class = default_filter_class
        self.log(
            f"当前列表无合格助战，刷新 {self.support_refreshes}/{max_refreshes}"
        )
        self._tap(point, "刷新助战列表")

    def _scroll_support_page(
        self,
        screen: np.ndarray,
        reason: str,
        max_scrolls: int,
    ) -> None:
        """Swipe the support list down one page and verify it moved."""
        settings = self.config.get("support", {})
        start = self._scaled_point(
            settings.get("scroll_start", [1300, 790]),
            screen,
        )
        step, row_pitch = self._support_scroll_step(screen, settings)
        end = (start[0], max(0, start[1] - step))
        self.log(
            f"{reason}，尝试向下翻页 "
            f"{self.support_scrolls + 1}/{max_scrolls}；"
            + (
                f"检测到卡片行距约{row_pitch}px，"
                f"使用保留重叠的{step}px手势引入两行新候选"
                if row_pitch is not None
                else f"使用保留重叠的{step}px手势翻页"
            )
        )
        if self.dry_run:
            self.support_scrolls += 1
            return

        self.device.swipe(
            start[0],
            start[1],
            end[0],
            end[1],
            int(settings.get("scroll_duration_ms", 350)),
        )
        self._wait_interruptibly(
            float(settings.get("scroll_settle_seconds", 0.8))
        )
        moved_screen = self.device.capture()
        visual_change = self._support_list_visual_change(
            screen,
            moved_screen,
        )
        minimum_change = float(
            settings.get("scroll_min_visual_change", 4.0)
        )

        if visual_change < minimum_change:
            retry_start = self._scaled_point(
                settings.get("scroll_retry_start", [760, 780]),
                moved_screen,
            )
            retry_end = (
                retry_start[0],
                max(0, retry_start[1] - step),
            )
            self.log(
                "首次滑动没有带动助战列表，"
                "改用列表中央快速上划重试"
            )
            self.device.swipe(
                retry_start[0],
                retry_start[1],
                retry_end[0],
                retry_end[1],
                int(
                    settings.get(
                        "scroll_retry_duration_ms",
                        260,
                    )
                ),
            )
            self._wait_interruptibly(
                float(settings.get("scroll_settle_seconds", 0.8))
            )
            retry_screen = self.device.capture()
            visual_change = self._support_list_visual_change(
                screen,
                retry_screen,
            )

        if visual_change >= minimum_change:
            self.support_scrolls += 1
            self.support_scroll_stalls = 0
            self.log(
                f"助战列表已实际移动（画面变化 {visual_change:.1f}），"
                f"继续检查第 {self.support_scrolls + 1} 屏"
            )
        else:
            self.support_scroll_stalls += 1
            max_stalls = int(
                settings.get("max_scroll_stalls", 2)
            )
            if self.support_scroll_stalls >= max_stalls:
                self.support_scrolls = max_scrolls
                self.log(
                    "连续滑动后助战列表仍未移动，"
                    "按已到列表底部处理，下一轮继续搜索或刷新"
                )
            else:
                self.log(
                    "本次滑动未生效且不计入已检查页数，"
                    f"下一轮继续重试（{self.support_scroll_stalls}/"
                    f"{max_stalls}）"
                )

    def _select_highest_level_support(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> None:
        """Highest-level mode: borrow the best visible row right away.

        Skips the banded class/level search entirely and uses whichever
        list the game currently has open.  The visible row with the
        highest level number wins; NP5 rows beat same-level rows.  The
        list is never scrolled or refreshed: when no row level can be
        read, the topmost row is still borrowed directly.
        """
        selector = self.support_selector
        chooser = getattr(selector, "choose_highest_level_np", None)
        choice = chooser(screen) if callable(chooser) else None
        if choice is None:
            guest_method = getattr(
                selector,
                "guest_support_row_ranges",
                None,
            )
            if callable(guest_method):
                try:
                    guest_rows = guest_method(
                        screen,
                        respect_exclusion=False,
                    )
                except TypeError:
                    guest_rows = guest_method(screen)
            else:
                guest_rows = []
            forced_only_method = getattr(
                selector,
                "is_forced_only_screen",
                None,
            )
            normal_method = getattr(
                selector,
                "has_normal_support_row",
                None,
            )
            forced_only = (
                bool(guest_rows)
                and callable(forced_only_method)
                and bool(forced_only_method(screen))
                and callable(normal_method)
                and not bool(normal_method(screen))
            )
            if forced_only:
                self.log(
                    "等级最高模式：当前页只有客将助战，直接选择客将"
                )
                self._select_guest_after_full_scan(
                    screen,
                    match,
                    guest_rows,
                    scan_label="等级最高模式客将页",
                )
                return
            self._pause(
                "等级最高模式：当前界面没有可点击的普通助战行",
                screen,
                match,
            )

        self.selected_support = choice
        self.support_refreshes = 0
        self.support_scrolls = 0
        self.support_scroll_stalls = 0
        candidate = choice.candidate
        if int(candidate.get("level", 0)) > 0:
            properties = [f"{int(candidate.get('level', 0))}级"]
            if int(candidate.get("np_level", 0)) == 5:
                properties.append("宝具5")
            else:
                properties.append("未确认宝具5")
        else:
            properties = ["未读取等级数字"]
        self.log(
            "等级最高模式选择助战："
            + " · ".join(properties)
            + f" · 匹配 {choice.match.score:.3f}"
        )
        self._tap(choice.match.center, "等级最高的助战")

    def _restricted_party_slot_visually_filled(
        self,
        screen: np.ndarray,
        restriction_match: MatchResult,
    ) -> bool:
        """Distinguish a selected servant card from the empty restriction slot.

        The compact “编队限制” banner remains visible after a support has been
        selected, so the entry template alone cannot describe the state.  The
        area immediately above that banner is almost black on an empty slot,
        but contains a bright, detailed portrait on a filled servant card.
        """
        behavior = self.config.get("behavior", {})
        height, width = screen.shape[:2]
        base = self.config["screen"]
        scale_x = width / float(base["base_width"])
        scale_y = height / float(base["base_height"])
        half_width = round(
            float(
                behavior.get(
                    "restricted_party_filled_portrait_half_width",
                    110,
                )
            )
            * scale_x
        )
        top = round(
            float(
                behavior.get(
                    "restricted_party_filled_portrait_top",
                    180,
                )
            )
            * scale_y
        )
        bottom_gap = round(
            float(
                behavior.get(
                    "restricted_party_filled_portrait_bottom_gap",
                    35,
                )
            )
            * scale_y
        )
        x1 = max(0, restriction_match.center[0] - half_width)
        x2 = min(width, restriction_match.center[0] + half_width)
        y1 = max(0, top)
        y2 = min(height, restriction_match.center[1] - bottom_gap)
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        saturated_ratio = float(
            np.count_nonzero(
                hsv[:, :, 1]
                >= int(
                    behavior.get(
                        "restricted_party_filled_min_saturation",
                        70,
                    )
                )
            )
        ) / float(hsv.shape[0] * hsv.shape[1])
        bright_ratio = float(
            np.count_nonzero(
                hsv[:, :, 2]
                >= int(
                    behavior.get(
                        "restricted_party_filled_min_value",
                        120,
                    )
                )
            )
        ) / float(hsv.shape[0] * hsv.shape[1])
        edge_ratio = float(
            np.count_nonzero(cv2.Canny(gray, 60, 140))
        ) / float(gray.size)

        filled = (
            bright_ratio
            >= float(
                behavior.get(
                    "restricted_party_filled_min_bright_ratio",
                    0.15,
                )
            )
            and edge_ratio
            >= float(
                behavior.get(
                    "restricted_party_filled_min_edge_ratio",
                    0.025,
                )
            )
        )
        if filled:
            self.log(
                "受限助战槽位视觉复核为已选定："
                f"彩色 {saturated_ratio:.3f}、亮部 {bright_ratio:.3f}、"
                f"边缘 {edge_ratio:.3f}"
            )
        return filled

    def _execute(
        self,
        screen: np.ndarray,
        match: MatchResult,
    ) -> None:
        rule = match.rule
        if str(rule.get("name", "")) == "battle_attack":
            recovered_cards = self._command_card_recovery_match(screen)
            if recovered_cards is not None:
                self.log(
                    "Attack规则同时命中疑似指令卡过渡帧，"
                    f"恢复分数 {recovered_cards.score:.3f}；"
                    "改为继续选卡，禁止点击右下角“返回”"
                )
                match = recovered_cards
                rule = match.rule
        action = rule["action"]
        rule_name = str(rule.get("name", ""))
        if (
            rule_name == "command_cards"
            and self._skill_target_overlay_visible(screen)
        ):
            # 技能选人界面会误中指令卡模板（实测约 0.95 分）；此时按
            # 选人处理，点击当前输出位，避免在错误界面反复点卡。
            self.log(
                "选人界面特征组合命中，指令卡模板疑似假阳性；"
                "改为选择技能目标"
            )
            action = "target_damage_dealer"
        if rule_name == "party_start":
            select_rule = next(
                (
                    candidate
                    for candidate in self.config.get("rules", [])
                    if candidate.get("name")
                    == "restricted_party_select_marker"
                ),
                None,
            )
            pending_select = (
                self.matcher.match_rule(screen, select_rule)
                if select_rule is not None
                else None
            )
            if pending_select is not None:
                self.log(
                    "战斗开始按钮已经出现，但受限队伍仍有“选择”空位；"
                    "改为自动编队，不尝试手动选人"
                )
                action = "restricted_party_setup"
        if rule_name == "skill_unavailable_close" and self._pending_np_slots:
            self._block_pending_np_slots(
                "宝具点击后出现不可用提示",
                screen,
            )
        elif rule_name == "command_cards" and self._pending_np_slots:
            self._block_pending_np_slots(
                "三次选卡后仍停留在指令卡界面",
                screen,
            )
        if rule.get("name") in {
            "battle_attack",
            "command_cards",
            "skill_unavailable_close",
            "battle_detail_close",
            "skill_target_support",
        }:
            self._battle_flow_active = True
        if rule_name == "free_quest_node":
            if self.farm_max_rounds > 0 and self.farm_round >= self.farm_max_rounds:
                self.log(
                    f"已达设定刷本次数 {self.farm_max_rounds} 次，"
                    "自动停止自动推进"
                )
                self.farm_limit_reached = True
                self.stop()
                return
            # 刷本：每点一次自由本节点计一轮，并记录详情条待"开始"。
            self.farm_round += 1
            self._farm_node_tapped = True
            self._farm_confirm_pending = False
            self.log(f"刷本第 {self.farm_round} 轮：点击自由本节点")
        elif rule_name == "free_quest_start":
            # 点过"开始任务"后清标志；抑制窗口防确认框残留帧重复点击；
            # 接下来等待出击确认弹窗（farm_confirm）。
            self._farm_node_tapped = False
            self._farm_strip_suppress_until = time.monotonic() + 2.0
            self._farm_confirm_pending = True
        elif rule_name != "farm_confirm":
            # 其他规则命中说明流程已越过确认弹窗（进入助战/编队/战斗），
            # 清除待确认标志。
            self._farm_confirm_pending = False
        self.log(
            f"识别 {rule['name']}，置信度 {match.score:.3f}，动作 {action}"
        )

        if self.dry_run:
            path = self._snapshot(screen, f"DRY-{rule['name']}", match)
            self.log(f"试运行不点击；识别截图：{path}")
            self.stop()
            return

        if action == "pause":
            self._pause(
                str(rule.get("reason", f"检测到 {rule['name']}")),
                screen,
                match,
            )
        elif action == "farm_confirm":
            point = self._scaled_point(
                self.config.get("behavior", {}).get(
                    "farm_confirm_point",
                    [1050, 712],
                ),
                screen,
            )
            self.log("确认出击弹窗：点击开始任务")
            self._tap(point, "确认出击开始任务")
        elif action == "close_free_quest":
            self._close_free_quest(screen, match)
        elif action == "tap_match_until_gone":
            self._tap_match_until_gone(screen, match)
        elif action == "tap_match":
            offset = rule.get("offset", [0, 0])
            scaled_offset = self._scaled_point(offset, screen)
            point = (
                match.center[0] + scaled_offset[0],
                match.center[1] + scaled_offset[1],
            )
            self._tap(point, rule["name"])
        elif action == "tap_point":
            self._tap(self._scaled_point(rule["point"], screen), rule["name"])
        elif action == "restricted_party_setup":
            select_rule = next(
                (
                    candidate
                    for candidate in self.config["rules"]
                    if candidate.get("name")
                    == "restricted_party_select_marker"
                ),
                None,
            )
            has_select_slot = (
                select_rule is not None
                and self.matcher.match_rule(screen, select_rule) is not None
            )
            filled_rule = next(
                (
                    candidate
                    for candidate in self.config["rules"]
                    if candidate.get("name")
                    == "restricted_party_filled_marker"
                ),
                None,
            )
            has_filled_restriction = (
                filled_rule is not None
                and self.matcher.match_rule(screen, filled_rule) is not None
            )
            if not has_filled_restriction:
                has_filled_restriction = (
                    self._restricted_party_slot_visually_filled(
                        screen,
                        match,
                    )
                )
            if has_select_slot:
                # 【临时禁用】自动编队点击：保留代码备用，现阶段不生效。
                # point = self.config.get("behavior", {}).get(
                #     "restricted_party_auto_formation_point",
                #     [370, 850],
                # )
                # self.log(
                #     "检测到编队限制与待选择空位并存；"
                #     "优先使用自动编队完成受限队伍"
                # )
                # self._tap(
                #     self._scaled_point(point, screen),
                #     "受限队伍自动编队",
                # )
                self.log(
                    "检测到编队限制与待选择空位；自动编队已临时禁用，"
                    "不处理空位，直接开始战斗"
                )
            if has_filled_restriction or has_select_slot:
                party_start_rule = next(
                    (
                        candidate
                        for candidate in self.config["rules"]
                        if candidate.get("name") == "party_start"
                    ),
                    None,
                )
                party_start_match = (
                    self.matcher.match_rule(screen, party_start_rule)
                    if party_start_rule is not None
                    else None
                )
                if party_start_match is not None:
                    point = party_start_match.center
                else:
                    point = self._scaled_point(
                        self.config.get("behavior", {}).get(
                            "restricted_party_start_point",
                            [1467, 840],
                        ),
                        screen,
                    )
                self.log("结束编队并点击战斗开始")
                self._tap(point, "受限队伍战斗开始")
            else:
                offset = rule.get("offset", [0, 0])
                scaled_offset = self._scaled_point(offset, screen)
                self._tap(
                    (
                        match.center[0] + scaled_offset[0],
                        match.center[1] + scaled_offset[1],
                    ),
                    rule["name"],
                )
        elif action == "friend_request":
            if self._support_rule_enabled("auto_send_friend_request"):
                self._tap(match.center, "战后申请好友")
            else:
                end_point = self.config.get("support", {}).get(
                    "friend_request_end_point",
                    [412, 770],
                )
                self._tap(
                    self._scaled_point(end_point, screen),
                    "战后不申请好友，点击结束",
                )
        elif action == "story":
            mode = str(self.config["behavior"].get("story_mode", "skip"))
            if mode == "skip":
                choice_point = self._story_choice_point(screen)
                if choice_point is not None:
                    self.log(
                        "检测到跳过前的剧情选项，先选择第一项；"
                        "选项消失后再执行跳过"
                    )
                    self._tap(choice_point, "剧情选项第一项")
                else:
                    self._tap(match.center, "剧情跳过")
            elif mode == "advance":
                point = self.config["behavior"].get(
                    "story_advance_point",
                    [1480, 820],
                )
                self._tap(self._scaled_point(point, screen), "推进对话")
            else:
                self._pause(f"未知 story_mode：{mode}", screen, match)
        elif action == "quest_start":
            self._quest_start_action(screen, match)
        elif action == "select_support":
            self._select_support_action(screen, match)
        elif action == "select_forced_support":
            self._select_forced_support_action(screen, match)
        elif action == "support_unavailable":
            offset = rule.get("offset", [0, 0])
            scaled_offset = self._scaled_point(offset, screen)
            self.selected_support = None
            self._reset_support_class_search(reset_initial_page=False)
            self.support_refreshes += 1
            self._tap(
                (
                    match.center[0] + scaled_offset[0],
                    match.center[1] + scaled_offset[1],
                ),
                "退出失效助战并重试",
            )
        elif action == "target_damage_dealer":
            active_slots = (
                sorted(self._party_active_slots)
                if self._party_state_initialized
                and self._party_active_slots
                else self.battle_planner.active_party_slots(screen)
            )
            if not active_slots:
                self._pause(
                    "技能选人时没有可用的在场从者位置记录",
                    screen,
                    match,
                )
            if self._last_damage_slot in active_slots:
                slot = int(self._last_damage_slot)
            else:
                priority = [
                    int(value)
                    for value in self.config.get("battle", {}).get(
                        "damage_slot_priority",
                        [3, 1, 2],
                    )
                ]
                slot = next(
                    (
                        candidate
                        for candidate in priority
                        if candidate in active_slots
                    ),
                    active_slots[0],
                )
            point = self._skill_target_point(
                screen,
                slot,
                active_slots,
            )
            self._tap(point, f"单体强化选择当前输出位 {slot}")
            battle = self.config.get("battle", {})
            self._wait_with_skill_animation_tap(
                float(battle.get("after_skill_target_wait_seconds", 1.2)),
                screen,
                "单体强化选人后",
                start_delay_seconds=float(
                    battle.get(
                        "skill_animation_after_target_tap_delay_seconds",
                        0.08,
                    )
                ),
                animation_started_by_previous_tap=True,
            )
        elif action == "battle":
            self._battle_action(screen, match)
        else:
            self._pause(f"未知动作：{action}", screen, match)

        if rule.get("reset_battle", False):
            self.battle_turn = 0
            self._battle_flow_active = False
            self._single_survivor_skills_used = False
            self._aoe_np_fired = False
            self._post_np_charge_used = False
            self._used_skill_groups.clear()
            self._blocked_np_slots.clear()
            self._pending_np_slots.clear()
            self._last_damage_slot = None
            self._reset_party_state()
        if rule.get("reset_quest", False):
            self.enemy_class = None
            self.selected_support = None
            self.support_refreshes = 0
            self._reset_support_class_search()
        self.actions += 1
        # 完成一次真实动作说明状态机确实在前进，
        # 清零连续恢复计数，避免"故障→恢复→故障"的循环。
        self.selfcheck.note_healthy()
        if self.actions >= self.max_actions:
            self._pause(
                f"已达到单次最大动作数 {self.max_actions}",
                screen,
                match,
            )
        allow_repeat = bool(rule.get("allow_repeat", False))
        if action == "story" and self.config["behavior"].get("story_mode") == "advance":
            allow_repeat = True
        if not allow_repeat:
            self._blocked_rule = rule["name"]
            self._blocked_since = time.monotonic()
            self._blocked_fingerprint = self._screen_fingerprint(screen)
            self._blocked_center = match.center
        cooldown = float(rule.get("cooldown", 2.0))
        # 记录本帧执行的规则，下一帧优先复检：连续帧画面高度相关，
        # 优先命中后通常一次匹配即可返回，省掉其余规则的全量扫描。
        self._last_match_prefer = rule_name
        self._wait_interruptibly(cooldown)
        self._unknown_since = time.monotonic()
        self._current_unknown_limit = float(
            rule.get("unknown_pause_seconds", self.unknown_pause_seconds)
        )

    def run(self) -> None:
        mode = "试运行" if self.dry_run else "正式运行"
        self.log(f"启动：{mode}")
        self.log(
            f"已加载模板：{', '.join(self.matcher.available_rules) or '无'}"
        )
        if self.matcher.missing_rules:
            self.log(
                "未标定模板（相关规则不会生效）："
                f"{', '.join(sorted(self.matcher.missing_rules))}"
            )
        if not self.matcher.available_rules:
            raise PauseRequested("没有任何已标定模板，无法运行")
        if self.farm_mode and "free_quest_node" in self.matcher.missing_rules:
            # 刷本循环的核心步骤是点击自由本节点；模板未标定时循环
            # 永远无法进行，直接提示而不是默默暂停。
            self.log(
                "警告：刷本模式缺少 free_quest_node 模板"
                "（templates/free_quest_node.png 未标定）。"
                "请在地图界面运行 main.py calibrate --rule free_quest_node "
                "框选自由本节点，否则无法循环刷取。"
            )

        while not self.stop_event.is_set():
            try:
                screen = self._prefetcher.get()
            except AdbError:
                if self.stop_event.is_set():
                    return
                result = self.selfcheck.recover_adb("ADB 截图失败")
                if result is not None:
                    self._handle_recovered(
                        RecoveredAction("ADB 截图失败", result)
                    )
                    continue
                raise
            try:
                self._check_foreground(screen)
                if (
                    self._battle_rule_enabled("confirm_skill_use")
                    and self.battle_planner.skill_confirmation_visible(screen)
                ):
                    self._has_seen_recognized_rule = True
                    self._battle_flow_active = True
                    self._unknown_wait_context = None
                    if self.dry_run:
                        path = self._snapshot(
                            screen,
                            "DRY-skill_confirmation",
                        )
                        self.log(
                            f"试运行识别到技能确认框，不点击；截图：{path}"
                        )
                        self.stop()
                        continue
                    self.log("识别到遗留的技能使用确认框，自动处理")
                    # 复用战斗流程的处理：点“决定”；弹窗不关闭时
                    # 改点“返回”关闭弹窗，而不是在同一个位置反复点击。
                    self._resolve_skill_confirmation(
                        screen,
                        "主循环遗留技能弹窗",
                        accelerate_animation=False,
                    )
                    self._unknown_since = time.monotonic()
                    self._clear_blocked_rule()
                    continue

                fingerprint = self._screen_fingerprint(screen)
                static_repeat = (
                    self._last_static_none_fingerprint is not None
                    and self._fingerprint_similar(
                        fingerprint,
                        self._last_static_none_fingerprint,
                    )
                    # 连续跳过重匹配有上限，防止渐进变化被容差漏过。
                    and self._static_skip_count < 10
                )
                match = None
                if static_repeat:
                    # 画面与上一帧近似一致，且上一帧所有检测均未命中：
                    # 识别结果几乎必然相同，跳过匹配，直接按未知画面处理。
                    self._last_static_none_fingerprint = fingerprint
                    self._static_skip_count += 1
                else:
                    self._static_skip_count = 0
                    self._last_static_none_fingerprint = None
                    # Free quests are optional map nodes. Close their detail strip
                    # before the generic matcher can wait on an unknown screen, then
                    # resume searching the map for the main-quest "Next" marker.
                    match = self._free_quest_match(screen)
                    # 刷本：详情条"开始任务"之后的出击确认弹窗，模板分不够，
                    # 用组合特征识别并点固定坐标。
                    if match is None:
                        match = self._farm_confirm_match(screen)
                    # A forced story choice makes the top-right Skip button dim, so
                    # it cannot be reached through the normal white Skip template.
                    # Detect the choice panels as an independent high-priority state.
                    if match is None:
                        match = self._story_choice_match(screen)
                    if match is None:
                        # 剧情规则可见时优先复检，避免宽松阈值的其他规则
                        # （如 quest_confirm_start）在剧情弹窗上抢先乱点。
                        prefer: list[str] = self._story_visible_prefer(screen)
                        if (
                            self._last_match_prefer is not None
                            and self._last_match_prefer not in prefer
                        ):
                            prefer.append(self._last_match_prefer)
                        match = self.matcher.find_first(
                            screen,
                            exclude=self._current_exclude_rules(),
                            prefer=prefer or None,
                        )
                    if match is None:
                        # 记录静态未知帧指纹，下一帧画面相同时跳过识别。
                        self._last_static_none_fingerprint = fingerprint
                now = time.monotonic()

                if match is None:
                    self._clear_blocked_rule()
                    self._stable_name = None
                    self._stable_count = 0
                    (
                        unknown_limit,
                        unknown_context,
                        unknown_description,
                    ) = self._unknown_pause_policy(screen)
                    self._log_unknown_wait_context(
                        unknown_context,
                        unknown_description,
                        unknown_limit,
                    )
                    # 画面静止检测：连续 N 秒画面不变说明可能已卡死，
                    # 直接触发自检/暂停，不必等满超时。
                    static_limit = self._unknown_static_limit(unknown_context)
                    if (
                        self._unknown_static_fingerprint is not None
                        and self._fingerprint_similar(
                            fingerprint,
                            self._unknown_static_fingerprint,
                        )
                    ):
                        static_elapsed = now - self._unknown_static_since
                    else:
                        self._unknown_static_fingerprint = fingerprint
                        self._unknown_static_since = now
                        static_elapsed = 0.0
                    if (
                        static_limit is not None
                        and self._unknown_static_since > 0
                        and static_elapsed >= static_limit
                    ):
                        self._pause(
                            f"{unknown_description}画面静止 "
                            f"{static_elapsed:.1f} 秒，可能已卡死",
                            screen,
                        )
                    elapsed = now - self._unknown_since
                    if elapsed >= unknown_limit:
                        self._pause(
                            f"{unknown_description}持续 {elapsed:.1f} 秒，"
                            "等待人工处理",
                            screen,
                        )
                    self._poll_wait()
                    continue

                self._has_seen_recognized_rule = True
                self._unknown_wait_context = None
                self._unknown_since = now
                self._current_unknown_limit = self.unknown_pause_seconds
                self._reset_unknown_static()
                name = match.rule["name"]
                if self._blocked_rule is not None:
                    if name == self._blocked_rule:
                        changed, visual_change, center_change = (
                            self._blocked_screen_has_changed(screen, match)
                        )
                        if changed:
                            self.log(
                                f"画面已切换但仍是 {name}："
                                f"视觉变化 {visual_change:.1f}，"
                                f"图标位移 {center_change:.1f}；继续推进"
                            )
                            self._clear_blocked_rule()
                            self._stable_name = None
                            self._stable_count = 0
                        else:
                            stuck_seconds = float(
                                self.config["behavior"].get(
                                    "action_stuck_pause_seconds",
                                    20,
                                )
                            )
                            if now - self._blocked_since >= stuck_seconds:
                                self._pause(
                                    f"动作 {name} 后画面持续未变化，等待人工处理",
                                    screen,
                                    match,
                                )
                            self._poll_wait()
                            continue
                    else:
                        self._clear_blocked_rule()

                if name == self._stable_name:
                    self._stable_count += 1
                else:
                    self._stable_name = name
                    self._stable_count = 1

                required = int(match.rule.get("stable_frames", self.stable_frames))
                if self._stable_count < required:
                    self._poll_wait()
                    continue

                self._stable_name = None
                self._stable_count = 0
                self._execute(screen, match)
            except RecoveredAction as exc:
                self._handle_recovered(exc)
                continue
            except AdbError:
                if self.stop_event.is_set():
                    return
                result = self.selfcheck.recover_adb("ADB 操作失败")
                if result is not None:
                    self._handle_recovered(
                        RecoveredAction("ADB 操作失败", result)
                    )
                    continue
                raise

    def _handle_recovered(self, exc: RecoveredAction) -> None:
        """自检恢复成功后的状态复位，然后回到主循环继续推进。"""
        result = exc.result
        if result.restarted_app or result.restarted_emulator:
            self._reset_session_state()
        else:
            self._clear_blocked_rule()
            self._stable_name = None
            self._stable_count = 0
            self._unknown_since = time.monotonic()
            self._current_unknown_limit = self.unknown_pause_seconds
            self._unknown_wait_context = None
            self._last_static_none_fingerprint = None
            self._reset_unknown_static()
        if result.restarted_emulator:
            self._prefetcher.request()
        message = (
            f"已自动恢复：{exc.reason}；执行："
            f"{'、'.join(result.actions_taken) or '无'}；"
            f"连续恢复 {self.selfcheck.consecutive_recoveries} 次"
        )
        self.log(message)
        if self.on_recovered is not None:
            try:
                self.on_recovered(message)
            except Exception:
                pass

    def _reset_session_state(self) -> None:
        """重启游戏/模拟器后清空战斗与助战簿记，从头重新识别画面。

        游戏冷启动需要时间，重新给 startup 未知画面最长等待宽限。
        """
        self.battle_turn = 0
        self._battle_flow_active = False
        self._single_survivor_skills_used = False
        self._aoe_np_fired = False
        self._post_np_charge_used = False
        self._used_skill_groups.clear()
        self._blocked_np_slots.clear()
        self._pending_np_slots.clear()
        self._last_damage_slot = None
        self._reset_party_state()
        self.enemy_class = None
        self.selected_support = None
        self.support_refreshes = 0
        self._reset_support_class_search()
        self._clear_blocked_rule()
        self._stable_name = None
        self._stable_count = 0
        self._has_seen_recognized_rule = False
        self._unknown_since = time.monotonic()
        self._current_unknown_limit = self.startup_unknown_pause_seconds
        self._unknown_wait_context = None
        self._last_static_none_fingerprint = None
        self._reset_unknown_static()
        self._farm_node_tapped = False
        self._farm_confirm_pending = False
        self._last_match_prefer = None
