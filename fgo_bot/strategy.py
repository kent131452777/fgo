from __future__ import annotations

import base64
import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .class_affinity import (
    counter_classes as official_counter_classes,
    has_attack_advantage,
)
from .config import resolve_from_config
from .vision import read_image


@dataclass(frozen=True)
class VisualMatch:
    score: float
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


@dataclass(frozen=True)
class SupportChoice:
    candidate: dict[str, Any]
    match: VisualMatch
    enemy_class: str
    counter_class: str


@dataclass(frozen=True)
class SupportLevelReading:
    value: int
    maximum: int
    confidence: float
    match: VisualMatch


@dataclass(frozen=True)
class CardInfo:
    name: str
    point: tuple[int, int]
    color: str
    color_confidence: float
    affinity: str
    affinity_confidence: float
    owner_feature: np.ndarray
    is_support: bool = False
    support_confidence: float = 0.0
    critical_percent: int | None = None
    critical_confidence: float = 0.0
    is_disabled: bool = False
    disabled_confidence: float = 0.0


@dataclass(frozen=True)
class CardPlan:
    labels: list[str]
    reason: str
    colors: list[str]


class BattleVisionError(RuntimeError):
    pass


def _normalize_digit(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    normalized = np.zeros((28, 20), dtype=np.uint8)
    if xs.size == 0:
        return normalized
    glyph = mask[
        int(ys.min()) : int(ys.max()) + 1,
        int(xs.min()) : int(xs.max()) + 1,
    ]
    scale = min(16.0 / glyph.shape[1], 24.0 / glyph.shape[0])
    width = max(1, round(glyph.shape[1] * scale))
    height = max(1, round(glyph.shape[0] * scale))
    glyph = cv2.resize(
        glyph,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    x = (normalized.shape[1] - width) // 2
    y = (normalized.shape[0] - height) // 2
    normalized[y : y + height, x : x + width] = glyph
    return normalized


@lru_cache(maxsize=1)
def _digit_templates() -> tuple[tuple[int, np.ndarray], ...]:
    """Build a small font ensemble for the outlined FGO percentage digits."""
    templates: list[tuple[int, np.ndarray]] = []
    for font in range(8):
        for thickness in (1, 2, 3, 4):
            for scale in (0.7, 0.8, 0.9, 1.0, 1.1):
                for digit in range(10):
                    canvas = np.zeros((50, 40), dtype=np.uint8)
                    cv2.putText(
                        canvas,
                        str(digit),
                        (4, 38),
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA,
                    )
                    binary = np.where(canvas > 80, 255, 0).astype(
                        np.uint8
                    )
                    templates.append((digit, _normalize_digit(binary)))
    return tuple(templates)


# Normalized glyph samples taken from the outlined percentage font used by
# FGO command cards.  Each value is a packed 28x20 binary mask.  Rendering
# generic OS fonts was noticeably less reliable for the game's italic serif
# numerals, especially 2/3/6/9.
_CRITICAL_DIGIT_GLYPHS: dict[int, tuple[str, ...]] = {
    0: (
        "AAAAAAAAAAAAAAA8AA/gA8+APDgHg8DwPA8DweA8HgPD4Dw+A8PgPDwDg8A4AAQAAAAAAAHAwAwMAP+AAAAAAAAAAAAAAA==",
        "AAAAAAAAAAAAAAB+AB/wA8eAODgPg8DwPA8DweA8HgPD4Dw+A8PAPDwDg8B4AAAAAAAAAADAwA4cAH8AAAAAAAAAAAAAAA==",
        "AAAAAAAAAAAAAAAAAAfgAP8AHHgHg8B4PA8DwPA8HgPB4DweA8HgPD4DgcB4AAAABAAAAADgwAc4AAAAAAAAAAAAAAAAAA==",
    ),
    1: (
        "AAAAAAAAAQAA8AA/AAfwA/8A//AfvwHD4AA+AAPgAD4AA8AAPAAHwAB4AAeAAHgAB4AAAAAAAAAAAA8AAPAAHwAAAAAAAA==",
        "AAAAAAAADwAD8AA/AA/wB/8B//AfPgCD4AA+AAPgADwAA8AAfAAHwAB4AAeAAHgAAAAAAAAAAAAAAA8AAPAAHwAAAAAAAA==",
    ),
    2: (
        "AAAAAAAAAAAAAAAcAAf4AefAODwDA8AwPAADwAA8CAeAgHgADwABwAA4AAMAAAAAAAAAAAD/8B//AAAAAAAAAAAAAAAAAA==",
        "AAAAAAAAAAAAADg+Awf4McfDODwzA8MjPDBjwwQ8OMfD+Pg/DgPh4Dw4Q8cMMAHAAAAAAABgcA//A//wAAAAAAAAAAAAAA==",
    ),
    3: (
        "AAAAAAAAAAAAAAA+AA/4A+/AOHwDA8AAfAAHwAB4AA+AD+AA/wAB+AAHgAB4AAAAAAAAAAMA8DAeA/+AAAAAAAAAAAAAAA==",
        "AAAAAAAAAAAAAAB/AB/4A8fAMHwCA8AAfAAHwAB4AB8AD+AA/wAA+AAHgAB4AAeAAAAAAAMAYDAeA8fAAAAAAAAAAAAAAA==",
        "AAAAAAAAAAAAAAAeAA/4A//CPHwDA8AAfCAHwAB4AA+AD+AA/wAB+AAHgAB4AAAAAAAAAAEA8DAeAf/AAAAAAAAAAAAAAA==",
    ),
    4: (
        "AAAAAAAAAYAAOAAHggD4OB+Dwfg4P4CP+ADvgA74Ac8AefAHHwBx8BweA4HgOBwBAcAAAAAAAAAAAAHAABwAA8AAAAAAAA==",
        "AAAAAAAEAcAAfAAHwAD8AB/AA/gAP4AH+AD/gB74Ac8AOPAHjwDw8BwOA8DgOA4AAOAAAAAAAAAAAAHAADwAA8AAAAAAAA==",
    ),
    5: (
        "AAAAAAAAAAAAAAADwB/8Af/AP/gDgAAwAAMAAGAAB/wA//AAD4AA+AAHgAB4AAAAAAAAAAEAADg8AP8AAAAAAAAAAAAAAA==",
    ),
    6: (
        "AAAAAAAAAAPA/Dw/ww8cMeDCPAAHhgDwAA8AAPfwH/8B+Pw/B8PgPD4jw8I8PEOABAAAQAAAABwOAPPAA/gAAAAAAAAAAA==",
        "AAAAAAAAAAPA/Dw/ww8MMeBCPAAHjwDwAA8AAPfwH/+B+HwfB8PgPD4jwcI8HEPABAAAQAAAgAwOAP/AAAAAAAAAAAAAAA==",
    ),
    7: (
        "AAAAAAAAAAAAAAAAAH/8B//A//wAA4AAcAAHAAHAABgAA4AAMAAOAADAAAgAAAAAAAAGAADAADgAAAAAAAAAAAAAAAAAAA==",
        "AAAAAAAAAAAAAAAAAH/8B//A//gAA4AAcAAHAAHAABwAA4AAcAAGAADAAAgAAAAAAAAGAABgABgAAAAAAAAAAAAAAAAAAA==",
    ),
    9: (
        "AAAAAAAAAAAAAAA+AA/wA8eAfHwPA8HwPB8DwfA8HwPB8HwfD8D5/AP3gAB4AAAAAAAAAAABgDAwA/wAAAAAAAAAAAAAAA==",
    ),
}


@lru_cache(maxsize=1)
def _critical_digit_templates() -> tuple[tuple[int, np.ndarray], ...]:
    templates: list[tuple[int, np.ndarray]] = []
    for digit, encoded_masks in _CRITICAL_DIGIT_GLYPHS.items():
        for encoded in encoded_masks:
            packed = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            bits = np.unpackbits(packed)[: 28 * 20]
            templates.append(
                (digit, bits.reshape(28, 20).astype(np.uint8) * 255)
            )

    # No 80% sample was available in the calibration captures.  Keep a small
    # font ensemble for 8; its two-loop silhouette is distinctive, and a high
    # confidence gate below prevents uncertain glyphs from receiving a value.
    for font in (
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC,
    ):
        for thickness in (2, 3):
            canvas = np.zeros((50, 40), dtype=np.uint8)
            cv2.putText(
                canvas,
                "8",
                (4, 38),
                font,
                1.0,
                255,
                thickness,
                cv2.LINE_AA,
            )
            templates.append(
                (
                    8,
                    _normalize_digit(
                        np.where(canvas > 80, 255, 0).astype(np.uint8)
                    ),
                )
            )
    return tuple(templates)


def _scaled_template(
    template: np.ndarray,
    screen: np.ndarray,
    base_width: int,
    base_height: int,
) -> np.ndarray:
    height, width = screen.shape[:2]
    scale_x = width / base_width
    scale_y = height / base_height
    if abs(scale_x - 1) < 0.01 and abs(scale_y - 1) < 0.01:
        return template
    new_size = (
        max(2, round(template.shape[1] * scale_x)),
        max(2, round(template.shape[0] * scale_y)),
    )
    interpolation = cv2.INTER_AREA if scale_x < 1 else cv2.INTER_CUBIC
    return cv2.resize(template, new_size, interpolation=interpolation)


def _match_template(
    screen: np.ndarray,
    template: np.ndarray,
    *,
    base_width: int,
    base_height: int,
    roi: list[int] | None = None,
) -> VisualMatch:
    scaled = _scaled_template(template, screen, base_width, base_height)
    screen_height, screen_width = screen.shape[:2]
    offset_x = 0
    offset_y = 0
    source = screen
    if roi:
        scale_x = screen_width / base_width
        scale_y = screen_height / base_height
        x1, y1, x2, y2 = (
            round(roi[0] * scale_x),
            round(roi[1] * scale_y),
            round(roi[2] * scale_x),
            round(roi[3] * scale_y),
        )
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(screen_width, x2), min(screen_height, y2)
        source = screen[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    if (
        source.shape[0] < scaled.shape[0]
        or source.shape[1] < scaled.shape[1]
    ):
        return VisualMatch(-1.0, 0, 0, scaled.shape[1], scaled.shape[0])

    result = cv2.matchTemplate(
        cv2.cvtColor(source, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY),
        cv2.TM_CCOEFF_NORMED,
    )
    _, score, _, location = cv2.minMaxLoc(result)
    return VisualMatch(
        float(score),
        int(location[0] + offset_x),
        int(location[1] + offset_y),
        int(scaled.shape[1]),
        int(scaled.shape[0]),
    )


def _find_template_matches(
    screen: np.ndarray,
    template: np.ndarray,
    *,
    base_width: int,
    base_height: int,
    roi: list[int] | None,
    threshold: float,
    max_results: int = 8,
) -> list[VisualMatch]:
    scaled = _scaled_template(template, screen, base_width, base_height)
    screen_height, screen_width = screen.shape[:2]
    offset_x = 0
    offset_y = 0
    source = screen
    if roi:
        scale_x = screen_width / base_width
        scale_y = screen_height / base_height
        x1, y1, x2, y2 = (
            round(roi[0] * scale_x),
            round(roi[1] * scale_y),
            round(roi[2] * scale_x),
            round(roi[3] * scale_y),
        )
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(screen_width, x2), min(screen_height, y2)
        source = screen[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    if (
        source.shape[0] < scaled.shape[0]
        or source.shape[1] < scaled.shape[1]
    ):
        return []
    result = cv2.matchTemplate(
        cv2.cvtColor(source, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY),
        cv2.TM_CCOEFF_NORMED,
    )
    output: list[VisualMatch] = []
    height, width = scaled.shape[:2]
    for _ in range(max_results):
        _, score, _, location = cv2.minMaxLoc(result)
        if score < threshold:
            break
        x, y = int(location[0]), int(location[1])
        output.append(
            VisualMatch(
                float(score),
                x + offset_x,
                y + offset_y,
                width,
                height,
            )
        )
        left = max(0, x - width // 2)
        top = max(0, y - height // 2)
        right = min(result.shape[1], x + width + width // 2)
        bottom = min(result.shape[0], y + height + height // 2)
        result[top:bottom, left:right] = -1.0
    return output


def _find_mask_matches(
    screen: np.ndarray,
    template: np.ndarray,
    *,
    base_width: int,
    base_height: int,
    roi: list[int] | None,
    hsv_lower: list[int],
    hsv_upper: list[int],
    threshold: float,
    max_results: int = 8,
) -> list[VisualMatch]:
    scaled = _scaled_template(template, screen, base_width, base_height)
    screen_height, screen_width = screen.shape[:2]
    offset_x = 0
    offset_y = 0
    source = screen
    if roi:
        scale_x = screen_width / base_width
        scale_y = screen_height / base_height
        x1, y1, x2, y2 = (
            round(roi[0] * scale_x),
            round(roi[1] * scale_y),
            round(roi[2] * scale_x),
            round(roi[3] * scale_y),
        )
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(screen_width, x2), min(screen_height, y2)
        source = screen[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    if (
        source.shape[0] < scaled.shape[0]
        or source.shape[1] < scaled.shape[1]
    ):
        return []
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    source_mask = cv2.inRange(
        cv2.cvtColor(source, cv2.COLOR_BGR2HSV),
        lower,
        upper,
    )
    template_mask = cv2.inRange(
        cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV),
        lower,
        upper,
    )
    result = cv2.matchTemplate(
        source_mask,
        template_mask,
        cv2.TM_CCOEFF_NORMED,
    )
    output: list[VisualMatch] = []
    height, width = scaled.shape[:2]
    for _ in range(max_results):
        _, score, _, location = cv2.minMaxLoc(result)
        if score < threshold:
            break
        x, y = int(location[0]), int(location[1])
        output.append(
            VisualMatch(
                float(score),
                x + offset_x,
                y + offset_y,
                width,
                height,
            )
        )
        left = max(0, x - width // 2)
        top = max(0, y - height // 2)
        right = min(result.shape[1], x + width + width // 2)
        bottom = min(result.shape[0], y + height + height // 2)
        result[top:bottom, left:right] = -1.0
    return output


class SupportSelector:
    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
        self.config = config
        self.config_path = config_path
        self.settings = config.get("support", {})
        self.base_width = int(config["screen"]["base_width"])
        self.base_height = int(config["screen"]["base_height"])
        self.enemy_templates: dict[str, np.ndarray] = {}
        self.candidate_templates: dict[str, np.ndarray] = {}
        self.generic_level_template: np.ndarray | None = None
        self.generic_np5_template: np.ndarray | None = None
        self.guest_support_template: np.ndarray | None = None
        self.last_registered_rejections: list[str] = []

        for class_name, raw_path in self.settings.get(
            "enemy_class_templates", {}
        ).items():
            path = resolve_from_config(config_path, raw_path)
            if path.is_file():
                self.enemy_templates[class_name] = read_image(path)

        for candidate in self.settings.get("candidates", []):
            raw_path = candidate.get("template")
            if not raw_path:
                continue
            path = resolve_from_config(config_path, raw_path)
            if path.is_file():
                self.candidate_templates[str(candidate["id"])] = read_image(path)

        level_path = self.settings.get("generic_level120_template")
        if level_path:
            path = resolve_from_config(config_path, level_path)
            if path.is_file():
                self.generic_level_template = read_image(path)
        np5_path = self.settings.get("generic_np5_template")
        if np5_path:
            path = resolve_from_config(config_path, np5_path)
            if path.is_file():
                self.generic_np5_template = read_image(path)
        guest_path = self.settings.get("guest_support_template")
        if guest_path:
            path = resolve_from_config(config_path, guest_path)
            if path.is_file():
                self.guest_support_template = read_image(path)

    @property
    def qualified_candidate_count(self) -> int:
        return sum(
            1
            for candidate in self.settings.get("candidates", [])
            if self._is_qualified(candidate)
            and str(candidate.get("id")) in self.candidate_templates
        )

    def _is_qualified(self, candidate: dict[str, Any]) -> bool:
        options = self.settings.get("strategy_options", {})
        require_level = bool(options.get("require_level_120", True))
        require_np5 = bool(options.get("require_np5", True))
        return (
            bool(candidate.get("enabled", True))
            and (
                not require_level
                or int(candidate.get("level", 0))
                >= int(self.settings.get("min_servant_level", 120))
            )
            and (
                not require_np5
                or int(candidate.get("np_level", 0)) == 5
            )
        )

    def detect_enemy_class(self, screen: np.ndarray) -> tuple[str, float] | None:
        threshold = float(self.settings.get("enemy_class_threshold", 0.86))
        roi = self.settings.get("enemy_class_roi")
        best: tuple[str, float] | None = None
        for class_name, template in self.enemy_templates.items():
            match = _match_template(
                screen,
                template,
                base_width=self.base_width,
                base_height=self.base_height,
                roi=roi,
            )
            if match.score >= threshold and (
                best is None or match.score > best[1]
            ):
                best = class_name, match.score
        return best

    def counter_classes(self, enemy_class: str) -> list[str]:
        return official_counter_classes(
            enemy_class,
            self.settings.get("counter_map", {}),
        )

    def desired_classes(self, enemy_class: str | None) -> list[str]:
        options = self.settings.get("strategy_options", {})
        if (
            enemy_class
            and bool(options.get("prefer_class_advantage", True))
        ):
            return self.counter_classes(enemy_class)
        if not bool(options.get("use_berserker_fallback", True)):
            return []
        fallback = self.settings.get("unknown_enemy_class", "berserker")
        if isinstance(fallback, str):
            return [fallback]
        return [str(value) for value in fallback]

    def choose(
        self,
        screen: np.ndarray,
        enemy_class: str | None,
    ) -> SupportChoice | None:
        counters = self.desired_classes(enemy_class)
        if enemy_class and not counters:
            return None
        return self._choose_registered_candidates(
            screen,
            allowed_classes=counters,
            enemy_class_label=enemy_class or "unknown",
            require_unknown_opt_in=enemy_class is None,
        )

    def choose_from_current_filtered_list(
        self,
        screen: np.ndarray,
    ) -> SupportChoice | None:
        """Trust the game's recommended class filter and rank visible candidates."""
        return self._choose_registered_candidates(
            screen,
            allowed_classes=None,
            enemy_class_label="system_recommended",
            require_unknown_opt_in=False,
        )

    def selected_support_filter(self, screen: np.ndarray) -> str | None:
        """Return the class/filter tab that is already highlighted by the game."""
        points = dict(self.settings.get("class_filter_points", {}))
        recommended_point = self.settings.get("system_recommended_filter_point")
        if recommended_point is not None:
            points["system_recommended"] = recommended_point
        if not points:
            return None

        lower = np.array(
            self.settings.get(
                "selected_filter_hsv_lower",
                [85, 120, 140],
            ),
            dtype=np.uint8,
        )
        upper = np.array(
            self.settings.get(
                "selected_filter_hsv_upper",
                [110, 255, 255],
            ),
            dtype=np.uint8,
        )
        raw_strips = self.settings.get(
            "selected_filter_border_strips",
            [
                [-40, 25, 40, 50],
                [-42, -50, -28, 45],
                [28, -50, 42, 45],
            ],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        ranked: list[tuple[float, str]] = []
        for raw_name, raw_point in points.items():
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                continue
            center_x = round(float(raw_point[0]) * scale_x)
            center_y = round(float(raw_point[1]) * scale_y)
            score = 0.0
            for strip in raw_strips:
                x1 = max(0, center_x + round(float(strip[0]) * scale_x))
                y1 = max(0, center_y + round(float(strip[1]) * scale_y))
                x2 = min(
                    screen.shape[1],
                    center_x + round(float(strip[2]) * scale_x),
                )
                y2 = min(
                    screen.shape[0],
                    center_y + round(float(strip[3]) * scale_y),
                )
                roi = screen[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                marker = cv2.inRange(
                    cv2.cvtColor(roi, cv2.COLOR_BGR2HSV),
                    lower,
                    upper,
                )
                score += float(np.count_nonzero(marker)) / float(marker.size)
            ranked.append((score, str(raw_name)))
        if not ranked:
            return None
        ranked.sort(reverse=True)
        minimum = float(
            self.settings.get("selected_filter_min_score", 0.30)
        )
        margin = float(
            self.settings.get("selected_filter_min_margin", 0.06)
        )
        best_score, best_name = ranked[0]
        next_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < minimum or best_score - next_score < margin:
            return None
        return best_name

    @staticmethod
    def _merge_nearby_matches(
        matches: list[VisualMatch],
        *,
        y_tolerance: int = 24,
    ) -> list[VisualMatch]:
        merged: list[VisualMatch] = []
        for match in sorted(matches, key=lambda item: item.score, reverse=True):
            if any(
                abs(match.center[1] - kept.center[1]) <= y_tolerance
                for kept in merged
            ):
                continue
            merged.append(match)
        return sorted(merged, key=lambda item: item.center[1])

    def _level120_matches(self, screen: np.ndarray) -> list[VisualMatch]:
        if self.generic_level_template is None:
            return []
        roi = self.settings.get("candidate_roi", [0, 180, 1600, 890])
        mask_matches = _find_mask_matches(
            screen,
            self.generic_level_template,
            base_width=self.base_width,
            base_height=self.base_height,
            roi=roi,
            hsv_lower=self.settings.get(
                "generic_level120_hsv_lower",
                [0, 0, 190],
            ),
            hsv_upper=self.settings.get(
                "generic_level120_hsv_upper",
                [179, 25, 255],
            ),
            threshold=float(
                self.settings.get("generic_level120_threshold", 0.84)
            ),
            max_results=int(
                self.settings.get("generic_level120_max_matches", 12)
            ),
        )
        # Highlighted rows tint the white digits blue, while ordinary rows
        # keep them nearly white. The grayscale pass covers the highlighted
        # variant; the strict HSV pass avoids confusing 100/100 with 120/120.
        gray_matches = _find_template_matches(
            screen,
            self.generic_level_template,
            base_width=self.base_width,
            base_height=self.base_height,
            roi=roi,
            threshold=float(
                self.settings.get(
                    "generic_level120_gray_threshold",
                    0.60,
                )
            ),
            max_results=int(
                self.settings.get("generic_level120_max_matches", 12)
            ),
        )
        return self._merge_nearby_matches(mask_matches + gray_matches)

    def _level100_plus_matches(
        self,
        screen: np.ndarray,
    ) -> list[VisualMatch]:
        """Find the common ``等级1xx`` prefix used by level 100-120 rows."""
        if self.generic_level_template is None:
            return []
        raw_crop = self.settings.get(
            "generic_level100_prefix_crop",
            [0, 0, 50, 50],
        )
        x1, y1, x2, y2 = (
            max(0, int(raw_crop[0])),
            max(0, int(raw_crop[1])),
            min(self.generic_level_template.shape[1], int(raw_crop[2])),
            min(self.generic_level_template.shape[0], int(raw_crop[3])),
        )
        prefix = self.generic_level_template[y1:y2, x1:x2]
        if prefix.size == 0:
            return []
        matches = _find_template_matches(
            screen,
            prefix,
            base_width=self.base_width,
            base_height=self.base_height,
            roi=self.settings.get(
                "generic_level100_roi",
                [40, 180, 280, 890],
            ),
            threshold=float(
                self.settings.get("generic_level100_threshold", 0.74)
            ),
            max_results=int(
                self.settings.get("generic_level100_max_matches", 12)
            ),
        )
        return self._merge_nearby_matches(matches)

    @staticmethod
    def _recognize_support_digit(mask: np.ndarray) -> tuple[int, float]:
        glyph = _normalize_digit(mask)
        glyph_on = glyph > 0
        glyph_count = int(np.count_nonzero(glyph_on))
        best_digit = -1
        best_score = 0.0
        for digit, template in _digit_templates():
            template_on = template > 0
            denominator = glyph_count + int(np.count_nonzero(template_on))
            if denominator == 0:
                continue
            score = (
                2.0
                * float(np.count_nonzero(glyph_on & template_on))
                / float(denominator)
            )
            if score > best_score:
                best_digit = digit
                best_score = score
        return best_digit, best_score

    def _support_level_components(
        self,
        screen: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> list[tuple[int, int, int, int, int, float]]:
        """Extract digit-sized components from a strip of the screen.

        Two HSV passes are combined: the strict white pass keeps the
        ordinary rows, a looser pass catches dimmer glyphs on highlighted
        rows.  Each component carries its precomputed digit recognition.
        """
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        strict = np.where(
            (
                hsv[:, :, 1]
                <= int(self.settings.get("support_level_max_saturation", 100))
            )
            & (
                hsv[:, :, 2]
                >= int(self.settings.get("support_level_min_value", 145))
            ),
            255,
            0,
        ).astype(np.uint8)
        loose = np.where(
            (
                hsv[:, :, 1]
                <= int(
                    self.settings.get(
                        "support_level_loose_max_saturation",
                        120,
                    )
                )
            )
            & (
                hsv[:, :, 2]
                >= int(
                    self.settings.get(
                        "support_level_loose_min_value",
                        115,
                    )
                )
            ),
            255,
            0,
        ).astype(np.uint8)
        mask = cv2.bitwise_or(strict, loose)
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components: list[tuple[int, int, int, int, int, float]] = []
        for raw_component in stats[1:]:
            x, y, width, height = (
                int(value) for value in raw_component[:4]
            )
            if not (
                round(5 * scale_x) <= width <= round(30 * scale_x)
                and round(10 * scale_y) <= height <= round(30 * scale_y)
            ):
                continue
            digit, confidence = self._recognize_support_digit(
                mask[y : y + height, x : x + width]
            )
            components.append(
                (x1 + x, y1 + y, width, height, digit, confidence)
            )
        return components

    def _parse_level_sequence(
        self,
        screen: np.ndarray,
        components: list[tuple[int, int, int, int, int, float]],
    ) -> SupportLevelReading | None:
        """Parse a ``value/maximum`` text from one row of glyph components.

        The separator slash is a narrow glyph the digit recognizer is not
        confident about.  Both digit groups are chosen as subsets of the
        glyphs left/right of the slash, so stray components (``Lv.``
        leftovers, card art) can be skipped; the parse that uses the most
        glyphs on one consistent line wins.  Short two-digit texts such
        as ``90/90`` are covered as well.
        """
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        minimum_digit_confidence = float(
            self.settings.get("support_level_digit_min_confidence", 0.70)
        )
        slash_confidence_cap = float(
            self.settings.get("support_level_slash_confidence_cap", 0.75)
        )
        slash_max_width = round(
            float(self.settings.get("support_level_slash_max_width", 13))
            * scale_x
        )
        line_tolerance = float(
            self.settings.get("support_level_line_tolerance", 8)
        ) * scale_y
        best: tuple[int, int, float, SupportLevelReading] | None = None
        for slash in components:
            if (
                slash[2] > slash_max_width
                or slash[4] < 0
                or slash[5] >= slash_confidence_cap
            ):
                continue
            left_pool = [
                item
                for item in components
                if item[0] + item[2] <= slash[0]
            ]
            right_pool = [
                item
                for item in components
                if item[0] >= slash[0] + slash[2]
            ]
            for left_size in (3, 2, 1):
                if len(left_pool) < left_size:
                    continue
                for left in itertools.combinations(left_pool, left_size):
                    ordered_left = sorted(left, key=lambda item: item[0])
                    for right_size in (3, 2, 1):
                        if len(right_pool) < right_size:
                            continue
                        for right in itertools.combinations(
                            right_pool, right_size
                        ):
                            ordered_right = sorted(
                                right, key=lambda item: item[0]
                            )
                            digits = list(left) + list(right)
                            if any(
                                item[4] < 0
                                or item[5] < minimum_digit_confidence
                                for item in digits
                            ):
                                continue
                            centers = [
                                item[1] + item[3] / 2.0 for item in digits
                            ]
                            if max(centers) - min(centers) > line_tolerance:
                                continue
                            value = int(
                                "".join(
                                    str(item[4]) for item in ordered_left
                                )
                            )
                            maximum = int(
                                "".join(
                                    str(item[4]) for item in ordered_right
                                )
                            )
                            if not (1 <= value <= 120 and 1 <= maximum <= 120):
                                continue
                            confidence = min(item[5] for item in digits)
                            key = (
                                left_size + right_size,
                                value,
                                confidence,
                            )
                            if best is not None and key <= best[:3]:
                                continue
                            reading = SupportLevelReading(
                                value=value,
                                maximum=maximum,
                                confidence=confidence,
                                match=VisualMatch(
                                    confidence,
                                    ordered_left[0][0],
                                    min(
                                        item[1] for item in components
                                    ),
                                    ordered_right[-1][0]
                                    + ordered_right[-1][2]
                                    - ordered_left[0][0],
                                    max(
                                        item[1] + item[3]
                                        for item in components
                                    )
                                    - min(item[1] for item in components),
                                ),
                            )
                            best = (key[0], key[1], key[2], reading)
        return best[3] if best is not None else None

    def _read_support_row_level(
        self,
        screen: np.ndarray,
        row_top: int,
        row_bottom: int,
    ) -> SupportLevelReading | None:
        """Read the level text of one support row from its fixed offsets."""
        raw_roi = self.settings.get(
            "support_level_value_roi",
            [130, 180, 270, 890],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, round(float(raw_roi[0]) * scale_x))
        x2 = min(screen.shape[1], round(float(raw_roi[2]) * scale_x))
        raw_offsets = self.settings.get(
            "support_level_row_text_offsets",
            [7, 45],
        )
        y1 = max(0, row_top + round(float(raw_offsets[0]) * scale_y))
        y2 = min(
            screen.shape[0],
            row_top + round(float(raw_offsets[1]) * scale_y),
        )
        if y2 <= y1 or x2 <= x1:
            return None
        components = self._support_level_components(screen, x1, y1, x2, y2)
        if not components:
            return None
        return self._parse_level_sequence(screen, components)

    def support_level_readings(
        self,
        screen: np.ndarray,
    ) -> list[SupportLevelReading]:
        """Read the servant level text shown on each visible support row.

        Rows display a ``value/maximum`` text (for example ``90/90`` or
        ``100/100``) at a fixed offset from the row top.  Parsing is
        scoped per row band from the yellow Support strip so two-digit
        levels and dimmer glyphs are read as well; when no row band is
        visible the whole list area is scanned with the legacy grouping.
        """
        rows = self.normal_support_row_ranges(screen)
        readings: list[SupportLevelReading] = []
        if rows:
            for top, bottom in rows:
                reading = self._read_support_row_level(screen, top, bottom)
                if reading is not None:
                    readings.append(reading)
            return readings

        raw_roi = self.settings.get(
            "support_level_value_roi",
            [130, 180, 270, 890],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, round(float(raw_roi[0]) * scale_x))
        y1 = max(0, round(float(raw_roi[1]) * scale_y))
        x2 = min(screen.shape[1], round(float(raw_roi[2]) * scale_x))
        y2 = min(screen.shape[0], round(float(raw_roi[3]) * scale_y))
        components = self._support_level_components(screen, x1, y1, x2, y2)
        if not components:
            return []
        center_tolerance = float(
            self.settings.get("support_level_center_tolerance", 3.5)
        ) * scale_y
        groups: list[
            list[tuple[int, int, int, int, int, float]]
        ] = []
        for component in sorted(
            components,
            key=lambda item: (item[1] + item[3] / 2.0, item[0]),
        ):
            center_y = component[1] + component[3] / 2.0
            for group in groups:
                group_center = sum(
                    item[1] + item[3] / 2.0 for item in group
                ) / len(group)
                if abs(center_y - group_center) <= center_tolerance:
                    group.append(component)
                    break
            else:
                groups.append([component])
        for group in groups:
            reading = self._parse_level_sequence(screen, group)
            if reading is not None:
                readings.append(reading)
        return sorted(readings, key=lambda item: item.match.center[1])

    def _support_level_readings_for_band(
        self,
        screen: np.ndarray,
        minimum_level: int,
        maximum_level: int,
    ) -> list[SupportLevelReading]:
        """Read a level band, with a strict 120/120 template fallback."""
        readings = self.support_level_readings(screen)
        if not (minimum_level <= 120 <= maximum_level):
            return readings

        row_tolerance = float(
            self.settings.get("support_level_row_tolerance", 24)
        ) * (screen.shape[0] / self.base_height)
        fallback_threshold = float(
            self.settings.get(
                "generic_level120_band_fallback_threshold",
                0.84,
            )
        )
        for match in self._level120_matches(screen):
            if match.score < fallback_threshold:
                continue
            if any(
                abs(reading.match.center[1] - match.center[1])
                <= row_tolerance
                for reading in readings
            ):
                continue
            readings.append(
                SupportLevelReading(
                    value=120,
                    maximum=120,
                    confidence=match.score,
                    match=match,
                )
            )
        return sorted(readings, key=lambda item: item.match.center[1])

    def _np5_matches(self, screen: np.ndarray) -> list[VisualMatch]:
        if self.generic_np5_template is None:
            return []
        return _find_mask_matches(
            screen,
            self.generic_np5_template,
            base_width=self.base_width,
            base_height=self.base_height,
            roi=self.settings.get("candidate_roi", [0, 180, 1600, 890]),
            hsv_lower=self.settings.get(
                "generic_np5_hsv_lower",
                [8, 80, 90],
            ),
            hsv_upper=self.settings.get(
                "generic_np5_hsv_upper",
                [45, 255, 255],
            ),
            threshold=float(
                self.settings.get("generic_np5_threshold", 0.84)
            ),
            max_results=int(
                self.settings.get("generic_np5_max_matches", 12)
            ),
        )

    def _registered_row_is_visually_qualified(
        self,
        match: VisualMatch,
        *,
        level_matches: list[VisualMatch],
        np5_matches: list[VisualMatch],
    ) -> bool:
        options = self.settings.get("strategy_options", {})
        require_level = bool(options.get("require_level_120", True))
        require_np5 = bool(options.get("require_np5", True))
        if require_level and self.generic_level_template is not None:
            min_delta, max_delta = self.settings.get(
                "registered_level_marker_delta_y",
                [80, 160],
            )
            if not any(
                float(min_delta)
                <= match.center[1] - level_match.center[1]
                <= float(max_delta)
                for level_match in level_matches
            ):
                return False
        if require_np5 and self.generic_np5_template is not None:
            min_delta, max_delta = self.settings.get(
                "registered_np5_marker_delta_y",
                [-20, 80],
            )
            if not any(
                float(min_delta)
                <= np5_match.center[1] - match.center[1]
                <= float(max_delta)
                for np5_match in np5_matches
            ):
                return False
        return True

    def _choose_registered_candidates(
        self,
        screen: np.ndarray,
        *,
        allowed_classes: list[str] | None,
        enemy_class_label: str,
        require_unknown_opt_in: bool,
    ) -> SupportChoice | None:
        threshold = float(self.settings.get("candidate_threshold", 0.84))
        roi = self.settings.get("candidate_roi")
        prefer_aoe = bool(
            self.settings.get("strategy_options", {}).get(
                "prefer_aoe_np",
                True,
            )
        )
        ranked: list[
            tuple[float, int, int, int, dict[str, Any], VisualMatch]
        ] = []
        self.last_registered_rejections = []
        guest_rows = self.guest_support_row_ranges(screen)
        level_matches = [
            match
            for match in self._level120_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        np5_matches = [
            match
            for match in self._np5_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        for candidate in self.settings.get("candidates", []):
            candidate_id = str(candidate.get("id", ""))
            template = self.candidate_templates.get(candidate_id)
            servant_class = str(candidate.get("class", ""))
            if (
                template is None
                or not self._is_qualified(candidate)
            ):
                continue
            # “优先群体宝具”需要跨越整个助战列表，而不只是比较当前
            # 可见的几行。非群体候选会在 Runner 完成全表群体搜索后，
            # 通过通用 120 级/宝具 5 降级轮次重新获得选择机会。
            if (
                prefer_aoe
                and str(candidate.get("np_target", "")).lower() != "aoe"
            ):
                continue
            if allowed_classes is not None:
                if servant_class not in allowed_classes:
                    continue
                class_rank = allowed_classes.index(servant_class)
            else:
                class_rank = 0
            if (
                require_unknown_opt_in
                and not bool(candidate.get("allow_unknown_enemy", False))
            ):
                continue
            candidate_threshold = float(
                candidate.get("threshold", threshold)
            )
            match_mode = str(
                candidate.get(
                    "match_mode",
                    self.settings.get("candidate_match_mode", "gray"),
                )
            )
            if match_mode == "hsv_mask":
                matches = _find_mask_matches(
                    screen,
                    template,
                    base_width=self.base_width,
                    base_height=self.base_height,
                    roi=roi,
                    hsv_lower=candidate.get(
                        "hsv_lower",
                        self.settings.get(
                            "candidate_hsv_lower",
                            [8, 80, 90],
                        ),
                    ),
                    hsv_upper=candidate.get(
                        "hsv_upper",
                        self.settings.get(
                            "candidate_hsv_upper",
                            [45, 255, 255],
                        ),
                    ),
                    threshold=candidate_threshold,
                    max_results=int(
                        self.settings.get("candidate_max_matches", 8)
                    ),
                )
            else:
                matches = _find_template_matches(
                    screen,
                    template,
                    base_width=self.base_width,
                    base_height=self.base_height,
                    roi=roi,
                    threshold=candidate_threshold,
                    max_results=int(
                        self.settings.get("candidate_max_matches", 8)
                    ),
                )
            visible_matches = [
                match
                for match in matches
                if not self._point_in_guest_rows(match.center, guest_rows)
            ]
            matches = [
                match
                for match in visible_matches
                if self._registered_row_is_visually_qualified(
                    match,
                    level_matches=level_matches,
                    np5_matches=np5_matches,
                )
            ]
            if visible_matches and not matches:
                self.last_registered_rejections.append(candidate_id)
            if not matches:
                continue
            match = max(matches, key=lambda item: item.score)
            priority = int(candidate.get("priority", 0))
            aoe_rank = int(
                prefer_aoe
                and str(candidate.get("np_target", "")).lower() == "aoe"
            )
            ranked.append(
                (
                    match.score,
                    -class_rank,
                    aoe_rank,
                    priority,
                    candidate,
                    match,
                )
            )

        if not ranked:
            return None
        _, _, _, _, candidate, match = max(
            ranked,
            key=lambda item: (item[1], item[2], item[3], item[0]),
        )
        servant_class = str(candidate["class"])
        return SupportChoice(
            candidate=candidate,
            match=match,
            enemy_class=enemy_class_label,
            counter_class=servant_class,
        )

    def recommended_classes(self, screen: np.ndarray) -> list[str]:
        """Return each class tab carrying an orange advantage arrow."""
        points = self.settings.get("class_filter_points", {})
        if not isinstance(points, dict):
            return []
        raw_offset = self.settings.get(
            "system_recommended_marker_offset",
            [-28, -48, 28, -9],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        lower = np.array(
            self.settings.get(
                "system_recommended_marker_hsv_lower",
                [3, 140, 140],
            ),
            dtype=np.uint8,
        )
        upper = np.array(
            self.settings.get(
                "system_recommended_marker_hsv_upper",
                [28, 255, 255],
            ),
            dtype=np.uint8,
        )
        minimum = float(
            self.settings.get(
                "system_recommended_class_min_ratio",
                0.03,
            )
        )
        # Berserker always shows a mixed red/blue relation marker on many
        # quests. It is appended separately as the final fallback class.
        fallback = str(
            self.settings.get("unknown_enemy_class", "berserker")
        )
        order = self.settings.get(
            "system_recommended_class_order",
            [
                "saber",
                "archer",
                "lancer",
                "rider",
                "caster",
                "assassin",
                "berserker",
            ],
        )
        output: list[str] = []
        for raw_name in order:
            name = str(raw_name)
            if name in {"all", fallback}:
                continue
            point = points.get(name)
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            px, py = float(point[0]), float(point[1])
            x1 = max(0, round((px + float(raw_offset[0])) * scale_x))
            y1 = max(0, round((py + float(raw_offset[1])) * scale_y))
            x2 = min(
                screen.shape[1],
                round((px + float(raw_offset[2])) * scale_x),
            )
            y2 = min(
                screen.shape[0],
                round((py + float(raw_offset[3])) * scale_y),
            )
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            marker = cv2.inRange(hsv, lower, upper)
            ratio = float(np.count_nonzero(marker)) / float(marker.size)
            if ratio >= minimum:
                output.append(name)
        return output

    def enemy_tendency_classes(self, screen: np.ndarray) -> list[str]:
        """Read the ordered enemy-class tendency icons on the support page.

        The game displays the major enemy as the first, larger icon.  To
        avoid maintaining a second class-template library, the class filter
        icons visible on the same page are used as live templates.
        """
        points = self.settings.get("class_filter_points", {})
        if not isinstance(points, dict):
            return []
        raw_region = self.settings.get(
            "enemy_tendency_region",
            [470, 0, 650, 110],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, round(float(raw_region[0]) * scale_x))
        y1 = max(0, round(float(raw_region[1]) * scale_y))
        x2 = min(screen.shape[1], round(float(raw_region[2]) * scale_x))
        y2 = min(screen.shape[0], round(float(raw_region[3]) * scale_y))
        search = screen[y1:y2, x1:x2]
        if search.size == 0:
            return []
        search_edges = cv2.Canny(
            cv2.cvtColor(search, cv2.COLOR_BGR2GRAY),
            40,
            120,
        )
        radius_x = max(
            8,
            round(
                float(
                    self.settings.get(
                        "enemy_tendency_filter_icon_radius",
                        16,
                    )
                )
                * scale_x
            ),
        )
        radius_y = max(
            8,
            round(
                float(
                    self.settings.get(
                        "enemy_tendency_filter_icon_radius",
                        16,
                    )
                )
                * scale_y
            ),
        )
        minimum_scale = float(
            self.settings.get("enemy_tendency_template_scale_min", 0.70)
        )
        maximum_scale = float(
            self.settings.get("enemy_tendency_template_scale_max", 1.05)
        )
        scale_steps = max(
            2,
            int(self.settings.get("enemy_tendency_template_scale_steps", 8)),
        )
        threshold = float(
            self.settings.get("enemy_tendency_class_threshold", 0.50)
        )
        candidates: list[
            tuple[float, str, tuple[float, float]]
        ] = []
        for raw_name, raw_point in points.items():
            class_name = str(raw_name)
            if class_name in {"all", "recommended"}:
                continue
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
                continue
            center_x = round(float(raw_point[0]) * scale_x)
            center_y = round(float(raw_point[1]) * scale_y)
            icon = screen[
                max(0, center_y - radius_y) : min(
                    screen.shape[0],
                    center_y + radius_y,
                ),
                max(0, center_x - radius_x) : min(
                    screen.shape[1],
                    center_x + radius_x,
                ),
            ]
            if icon.size == 0:
                continue
            icon_edges = cv2.Canny(
                cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY),
                40,
                120,
            )
            best_score = -1.0
            best_center = (0.0, 0.0)
            for template_scale in np.linspace(
                minimum_scale,
                maximum_scale,
                scale_steps,
            ):
                template = cv2.resize(
                    icon_edges,
                    None,
                    fx=float(template_scale),
                    fy=float(template_scale),
                    interpolation=cv2.INTER_AREA,
                )
                if (
                    template.shape[0] > search_edges.shape[0]
                    or template.shape[1] > search_edges.shape[1]
                    or min(template.shape[:2]) < 5
                ):
                    continue
                result = cv2.matchTemplate(
                    search_edges,
                    template,
                    cv2.TM_CCOEFF_NORMED,
                )
                _, score, _, location = cv2.minMaxLoc(result)
                if float(score) > best_score:
                    best_score = float(score)
                    best_center = (
                        location[0] + template.shape[1] / 2.0,
                        location[1] + template.shape[0] / 2.0,
                    )
            if best_score >= threshold:
                candidates.append(
                    (best_score, class_name, best_center)
                )

        # Reject two class labels that accidentally matched the same symbol.
        spacing = float(
            self.settings.get("enemy_tendency_min_icon_spacing", 18)
        )
        kept: list[tuple[float, str, tuple[float, float]]] = []
        for candidate in sorted(candidates, reverse=True):
            if any(
                np.hypot(
                    candidate[2][0] - existing[2][0],
                    candidate[2][1] - existing[2][1],
                )
                < spacing
                for existing in kept
            ):
                continue
            kept.append(candidate)
        kept.sort(key=lambda item: (item[2][1], item[2][0]))
        return [class_name for _, class_name, _ in kept]

    def prioritize_recommended_classes(
        self,
        screen: np.ndarray,
        recommended: list[str],
    ) -> tuple[list[str], list[str]]:
        """Put the counter class for the major boss before minor enemies."""
        tendency = self.enemy_tendency_classes(screen)
        counter_priority: list[str] = []
        for enemy_class in tendency:
            for counter_class in self.counter_classes(enemy_class):
                if counter_class not in counter_priority:
                    counter_priority.append(counter_class)
        original_order = {
            class_name: index
            for index, class_name in enumerate(recommended)
        }
        priority_order = {
            class_name: index
            for index, class_name in enumerate(counter_priority)
        }
        ordered = sorted(
            recommended,
            key=lambda class_name: (
                priority_order.get(
                    class_name,
                    len(priority_order) + original_order[class_name],
                ),
                original_order[class_name],
            ),
        )
        return ordered, tendency

    def has_system_class_recommendation(self, screen: np.ndarray) -> bool:
        """Detect the orange advantage arrows shown above class filters."""
        if self.recommended_classes(screen):
            return True
        raw_region = self.settings.get(
            "system_recommended_marker_region",
            [135, 115, 735, 150],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1, y1, x2, y2 = (
            max(0, round(float(raw_region[0]) * scale_x)),
            max(0, round(float(raw_region[1]) * scale_y)),
            min(screen.shape[1], round(float(raw_region[2]) * scale_x)),
            min(screen.shape[0], round(float(raw_region[3]) * scale_y)),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array(
            self.settings.get(
                "system_recommended_marker_hsv_lower",
                [3, 140, 140],
            ),
            dtype=np.uint8,
        )
        upper = np.array(
            self.settings.get(
                "system_recommended_marker_hsv_upper",
                [28, 255, 255],
            ),
            dtype=np.uint8,
        )
        marker = cv2.inRange(hsv, lower, upper)
        ratio = float(np.count_nonzero(marker)) / float(marker.size)
        return ratio >= float(
            self.settings.get(
                "system_recommended_marker_min_ratio",
                0.01,
            )
        )

    @staticmethod
    def _point_in_guest_rows(
        point: tuple[int, int],
        rows: list[tuple[int, int]],
    ) -> bool:
        return any(top <= point[1] <= bottom for top, bottom in rows)

    def guest_support_row_ranges(
        self,
        screen: np.ndarray,
        *,
        respect_exclusion: bool = True,
    ) -> list[tuple[int, int]]:
        options = self.settings.get("strategy_options", {})
        if (
            respect_exclusion
            and not bool(options.get("exclude_guest_support", True))
        ):
            return []
        if self.guest_support_template is None:
            return []
        matches = _find_template_matches(
            screen,
            self.guest_support_template,
            base_width=self.base_width,
            base_height=self.base_height,
            roi=self.settings.get(
                "guest_support_marker_roi",
                [250, 180, 650, 850],
            ),
            threshold=float(
                self.settings.get("guest_support_threshold", 0.90)
            ),
            max_results=int(
                self.settings.get("guest_support_max_rows", 4)
            ),
        )
        raw_offset = self.settings.get(
            "guest_support_row_y_offset",
            [-30, 230],
        )
        scale_y = screen.shape[0] / self.base_height
        rows = [
            (
                max(
                    0,
                    match.y + round(float(raw_offset[0]) * scale_y),
                ),
                min(
                    screen.shape[0],
                    match.y + round(float(raw_offset[1]) * scale_y),
                ),
            )
            for match in matches
        ]
        return rows

    def has_qualified_counter(self, enemy_class: str | None) -> bool:
        if enemy_class is None:
            counters = set(self.desired_classes(None))
            return any(
                self._is_qualified(candidate)
                and bool(candidate.get("allow_unknown_enemy", False))
                and str(candidate.get("class")) in counters
                and str(candidate.get("id")) in self.candidate_templates
                for candidate in self.settings.get("candidates", [])
            )
        counters = set(self.counter_classes(enemy_class))
        return any(
            self._is_qualified(candidate)
            and str(candidate.get("class")) in counters
            and str(candidate.get("id")) in self.candidate_templates
            for candidate in self.settings.get("candidates", [])
        )

    @property
    def has_generic_level_np_templates(self) -> bool:
        return (
            self.generic_level_template is not None
            and self.generic_np5_template is not None
        )

    def choose_registered_level_band(
        self,
        screen: np.ndarray,
        servant_class: str | None,
        minimum_level: int,
        maximum_level: int,
        *,
        require_np5: bool = True,
        require_aoe: bool = True,
    ) -> SupportChoice | None:
        """Choose a registered servant whose visible row is in a level band."""
        guest_rows = self.guest_support_row_ranges(screen)
        level_readings = [
            reading
            for reading in self._support_level_readings_for_band(
                screen,
                minimum_level,
                maximum_level,
            )
            if minimum_level <= reading.value <= maximum_level
            and not self._point_in_guest_rows(reading.match.center, guest_rows)
        ]
        if not level_readings:
            return None
        np5_matches = [
            match
            for match in self._np5_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        if require_np5 and not np5_matches:
            return None

        threshold = float(self.settings.get("candidate_threshold", 0.84))
        roi = self.settings.get("candidate_roi")
        ranked: list[
            tuple[int, int, float, dict[str, Any], VisualMatch]
        ] = []
        for raw_candidate in self.settings.get("candidates", []):
            candidate_id = str(raw_candidate.get("id", ""))
            template = self.candidate_templates.get(candidate_id)
            candidate_class = str(raw_candidate.get("class", ""))
            if (
                template is None
                or not bool(raw_candidate.get("enabled", True))
                or (
                    servant_class is not None
                    and candidate_class != servant_class
                )
                or (
                    require_np5
                    and int(raw_candidate.get("np_level", 0)) != 5
                )
                or (
                    require_aoe
                    and str(raw_candidate.get("np_target", "")).lower()
                    != "aoe"
                )
            ):
                continue

            candidate_threshold = float(
                raw_candidate.get("threshold", threshold)
            )
            match_mode = str(
                raw_candidate.get(
                    "match_mode",
                    self.settings.get("candidate_match_mode", "gray"),
                )
            )
            if match_mode == "hsv_mask":
                matches = _find_mask_matches(
                    screen,
                    template,
                    base_width=self.base_width,
                    base_height=self.base_height,
                    roi=roi,
                    hsv_lower=raw_candidate.get(
                        "hsv_lower",
                        self.settings.get(
                            "candidate_hsv_lower",
                            [8, 80, 90],
                        ),
                    ),
                    hsv_upper=raw_candidate.get(
                        "hsv_upper",
                        self.settings.get(
                            "candidate_hsv_upper",
                            [45, 255, 255],
                        ),
                    ),
                    threshold=candidate_threshold,
                    max_results=int(
                        self.settings.get("candidate_max_matches", 8)
                    ),
                )
            else:
                matches = _find_template_matches(
                    screen,
                    template,
                    base_width=self.base_width,
                    base_height=self.base_height,
                    roi=roi,
                    threshold=candidate_threshold,
                    max_results=int(
                        self.settings.get("candidate_max_matches", 8)
                    ),
                )

            for match in matches:
                if self._point_in_guest_rows(match.center, guest_rows):
                    continue
                level_min, level_max = self.settings.get(
                    "registered_level_marker_delta_y",
                    [80, 160],
                )
                associated_levels = [
                    reading
                    for reading in level_readings
                    if float(level_min)
                    <= match.center[1] - reading.match.center[1]
                    <= float(level_max)
                ]
                if not associated_levels:
                    continue
                if require_np5:
                    np_min, np_max = self.settings.get(
                        "registered_np5_marker_delta_y",
                        [-20, 80],
                    )
                    if not any(
                        float(np_min)
                        <= np5_match.center[1] - match.center[1]
                        <= float(np_max)
                        for np5_match in np5_matches
                    ):
                        continue
                reading = max(
                    associated_levels,
                    key=lambda item: (item.value, item.confidence),
                )
                ranked.append(
                    (
                        reading.value,
                        int(raw_candidate.get("priority", 0)),
                        match.score,
                        raw_candidate,
                        match,
                    )
                )

        if not ranked:
            return None
        actual_level, _, _, raw_candidate, match = max(
            ranked,
            key=lambda item: (item[0], item[1], item[2]),
        )
        candidate = dict(raw_candidate)
        candidate["level"] = actual_level
        candidate["level_at_least"] = minimum_level
        candidate["relaxed_support"] = actual_level < 120
        return SupportChoice(
            candidate=candidate,
            match=match,
            enemy_class="system_default",
            counter_class=str(candidate.get("class", servant_class or "unknown")),
        )

    def choose_generic_level_band(
        self,
        screen: np.ndarray,
        servant_class: str,
        minimum_level: int,
        maximum_level: int,
        *,
        require_np5: bool = True,
    ) -> SupportChoice | None:
        """Choose a visibly levelled row when NP target type is not required."""
        guest_rows = self.guest_support_row_ranges(screen)
        level_readings = [
            reading
            for reading in self._support_level_readings_for_band(
                screen,
                minimum_level,
                maximum_level,
            )
            if minimum_level <= reading.value <= maximum_level
            and not self._point_in_guest_rows(reading.match.center, guest_rows)
        ]
        if not level_readings:
            return None
        np5_matches = [
            match
            for match in self._np5_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        if require_np5 and not np5_matches:
            return None
        min_delta, max_delta = self.settings.get(
            "generic_row_marker_delta_y",
            [80, 190],
        )
        ranked: list[
            tuple[int, float, SupportLevelReading, VisualMatch | None]
        ] = []
        for reading in level_readings:
            if require_np5:
                row_np5 = [
                    match
                    for match in np5_matches
                    if float(min_delta)
                    <= match.center[1] - reading.match.center[1]
                    <= float(max_delta)
                ]
                for np5_match in row_np5:
                    ranked.append(
                        (
                            reading.value,
                            reading.confidence + np5_match.score,
                            reading,
                            np5_match,
                        )
                    )
            else:
                ranked.append(
                    (
                        reading.value,
                        reading.confidence,
                        reading,
                        None,
                    )
                )
        if not ranked:
            return None
        actual_level, score, reading, np5_match = max(
            ranked,
            key=lambda item: (item[0], item[1]),
        )
        click_x = int(self.settings.get("generic_row_click_x", 700))
        if np5_match is not None:
            click_y = np5_match.center[1] + int(
                self.settings.get("generic_row_click_y_offset", -25)
            )
        else:
            click_y = reading.match.center[1] + int(
                self.settings.get("generic_level_click_y_offset", 95)
            )
        visual = VisualMatch(
            score=min(1.0, score / (2.0 if np5_match is not None else 1.0)),
            x=click_x - 1,
            y=click_y - 1,
            width=2,
            height=2,
        )
        candidate = {
            "id": f"generic_{servant_class}_lv{actual_level}",
            "name": f"{actual_level}级{'、宝具5' if require_np5 else ''} {servant_class}",
            "class": servant_class,
            "level": actual_level,
            "level_at_least": minimum_level,
            "np_level": 5 if require_np5 else 0,
            "np_target": "unknown",
            "np_damage": True,
            "np_color": "",
            "party_slot": int(
                self.settings.get("generic_party_slot", 3)
            ),
            "generic": True,
            "relaxed_support": actual_level < 120,
        }
        return SupportChoice(
            candidate=candidate,
            match=visual,
            enemy_class="system_default",
            counter_class=servant_class,
        )

    def choose_highest_level_np(
        self,
        screen: np.ndarray,
    ) -> SupportChoice | None:
        """Pick the best visible row on the current screen and borrow it.

        Highest-level mode ignores registered templates, class advantage
        and level bands.  Rows are ranked by (level descending, NP5 badge
        present, reading confidence); the winning row is tapped at its
        vertical center.  When no row level can be read the topmost row
        is still returned so the runner can borrow it directly without
        scrolling or refreshing.
        """
        guest_rows = self.guest_support_row_ranges(screen)
        rows = self.normal_support_row_ranges(screen)
        if not rows:
            return None
        readings = [
            reading
            for reading in self.support_level_readings(screen)
            if not self._point_in_guest_rows(reading.match.center, guest_rows)
        ]
        np5_matches = [
            match
            for match in self._np5_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]

        def reading_in_row(top: int, bottom: int) -> SupportLevelReading | None:
            return next(
                (
                    reading
                    for reading in readings
                    if top <= reading.match.center[1] <= bottom
                ),
                None,
            )

        def np5_in_row(top: int, bottom: int) -> bool:
            return any(
                top <= match.center[1] <= bottom
                for match in np5_matches
            )

        # Rows arrive top-to-bottom; the first maximum wins level ties.
        best: tuple[tuple[int, bool, float], int, int, SupportLevelReading | None, bool] | None = None
        for top, bottom in rows:
            reading = reading_in_row(top, bottom)
            np5_seen = np5_in_row(top, bottom)
            key = (
                reading.value if reading is not None else -1,
                np5_seen,
                reading.confidence if reading is not None else 0.0,
            )
            entry = (key, top, bottom, reading, np5_seen)
            if best is None or entry[0] > best[0]:
                best = entry
        assert best is not None
        _key, top, bottom, best_reading, np5_seen = best

        click_x = round(
            float(self.settings.get("generic_row_click_x", 700))
            * screen.shape[1]
            / self.base_width
        )
        click_y = (top + bottom) // 2
        if best_reading is not None:
            description = (
                f"{best_reading.value}级、宝具5"
                if np5_seen
                else f"{best_reading.value}级"
            )
            click_score = best_reading.confidence
            level_value = best_reading.value
        else:
            description = "未读取等级助战"
            click_score = 0.0
            level_value = 0
        visual = VisualMatch(
            score=click_score,
            x=click_x - 1,
            y=click_y - 1,
            width=2,
            height=2,
        )
        candidate = {
            "id": f"highest_{level_value}"
            + ("_np5" if np5_seen else ""),
            "name": description,
            "class": "unknown",
            "level": level_value,
            "level_at_least": level_value,
            "np_level": 5 if np5_seen else 0,
            "np_target": "unknown",
            "np_damage": True,
            "np_color": "",
            "party_slot": int(
                self.settings.get("generic_party_slot", 3)
            ),
            "generic": True,
            "relaxed_support": level_value < 120,
        }
        return SupportChoice(
            candidate=candidate,
            match=visual,
            enemy_class="highest_level",
            counter_class="highest_level",
        )

    def choose_generic_level120_np5(
        self,
        screen: np.ndarray,
        servant_class: str,
    ) -> SupportChoice | None:
        options = self.settings.get("strategy_options", {})
        require_level = bool(options.get("require_level_120", True))
        require_np5 = bool(options.get("require_np5", True))
        if require_level and self.generic_level_template is None:
            return None
        if require_np5 and self.generic_np5_template is None:
            return None
        if not require_level and not require_np5:
            return None
        guest_rows = self.guest_support_row_ranges(screen)
        level_matches = (
            self._level120_matches(screen)
            if require_level and self.generic_level_template is not None
            else []
        )
        level_matches = [
            match
            for match in level_matches
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        np5_matches = (
            self._np5_matches(screen)
            if require_np5 and self.generic_np5_template is not None
            else []
        )
        np5_matches = [
            match
            for match in np5_matches
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        min_delta, max_delta = self.settings.get(
            "generic_row_marker_delta_y",
            [80, 190],
        )
        pairs: list[tuple[float, VisualMatch, VisualMatch]] = []
        for level_match in level_matches:
            for np5_match in np5_matches:
                delta = np5_match.center[1] - level_match.center[1]
                if float(min_delta) <= delta <= float(max_delta):
                    pairs.append(
                        (
                            level_match.score + np5_match.score,
                            level_match,
                            np5_match,
                        )
                    )
        level_match: VisualMatch | None = None
        np5_match: VisualMatch | None = None
        if require_level and require_np5:
            if not pairs:
                return None
            _, level_match, np5_match = max(
                pairs,
                key=lambda item: (item[0], -item[1].center[1]),
            )
        elif require_level:
            if not level_matches:
                return None
            level_match = min(level_matches, key=lambda item: item.center[1])
        else:
            if not np5_matches:
                return None
            np5_match = min(np5_matches, key=lambda item: item.center[1])

        click_x = int(self.settings.get("generic_row_click_x", 700))
        if np5_match is not None:
            click_y = np5_match.center[1] + int(
                self.settings.get("generic_row_click_y_offset", -25)
            )
        else:
            assert level_match is not None
            click_y = level_match.center[1] + int(
                self.settings.get("generic_level_click_y_offset", 95)
            )
        scores = [
            match.score
            for match in (level_match, np5_match)
            if match is not None
        ]
        visual = VisualMatch(
            score=min(scores),
            x=click_x - 1,
            y=click_y - 1,
            width=2,
            height=2,
        )
        known_properties: list[str] = []
        if require_level:
            known_properties.append("120级")
        if require_np5:
            known_properties.append("宝具5")
        description = "、".join(known_properties)
        candidate = {
            "id": f"generic_{servant_class}_lv120_np5",
            "name": f"{description} {servant_class}",
            "class": servant_class,
            "level": 120 if require_level else 0,
            "np_level": 5 if require_np5 else 0,
            "np_target": str(
                self.settings.get("generic_np_target", "aoe")
            ),
            "np_damage": True,
            "np_color": str(
                self.settings.get("generic_np_color", "")
            ),
            "party_slot": int(
                self.settings.get("generic_party_slot", 3)
            ),
            "priority": 0,
            "generic": True,
        }
        return SupportChoice(
            candidate=candidate,
            match=visual,
            enemy_class="unknown",
            counter_class=servant_class,
        )

    def choose_registered_level100_np5_aoe(
        self,
        screen: np.ndarray,
        servant_class: str,
    ) -> SupportChoice | None:
        """Relax only the level-120 rule for a known NP5 AOE fallback."""
        threshold = float(self.settings.get("candidate_threshold", 0.84))
        guest_rows = self.guest_support_row_ranges(screen)
        level_matches = [
            match
            for match in self._level100_plus_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        np5_matches = [
            match
            for match in self._np5_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        ranked: list[tuple[int, float, dict[str, Any], VisualMatch]] = []
        for candidate in self.settings.get("candidates", []):
            candidate_id = str(candidate.get("id", ""))
            template = self.candidate_templates.get(candidate_id)
            if (
                template is None
                or not bool(candidate.get("enabled", True))
                or str(candidate.get("class", "")) != servant_class
                or int(candidate.get("np_level", 0)) != 5
                or str(candidate.get("np_target", "")).lower() != "aoe"
            ):
                continue
            matches = _find_template_matches(
                screen,
                template,
                base_width=self.base_width,
                base_height=self.base_height,
                roi=self.settings.get("candidate_roi"),
                threshold=float(candidate.get("threshold", threshold)),
                max_results=int(
                    self.settings.get("candidate_max_matches", 8)
                ),
            )
            for match in matches:
                if self._point_in_guest_rows(match.center, guest_rows):
                    continue
                if not self._registered_row_is_visually_level100_np5(
                    match,
                    level_matches=level_matches,
                    np5_matches=np5_matches,
                ):
                    continue
                ranked.append(
                    (
                        int(candidate.get("priority", 0)),
                        match.score,
                        candidate,
                        match,
                    )
                )
        if not ranked:
            return None
        _, _, raw_candidate, match = max(
            ranked,
            key=lambda item: (item[0], item[1]),
        )
        candidate = dict(raw_candidate)
        candidate["level"] = 100
        candidate["level_at_least"] = 100
        candidate["relaxed_support"] = True
        return SupportChoice(
            candidate=candidate,
            match=match,
            enemy_class="berserker_fallback",
            counter_class=servant_class,
        )

    def _registered_row_is_visually_level100_np5(
        self,
        match: VisualMatch,
        *,
        level_matches: list[VisualMatch],
        np5_matches: list[VisualMatch],
    ) -> bool:
        level_min, level_max = self.settings.get(
            "registered_level_marker_delta_y",
            [80, 160],
        )
        np_min, np_max = self.settings.get(
            "registered_np5_marker_delta_y",
            [-20, 80],
        )
        return (
            any(
                float(level_min)
                <= match.center[1] - level.center[1]
                <= float(level_max)
                for level in level_matches
            )
            and any(
                float(np_min)
                <= np5.center[1] - match.center[1]
                <= float(np_max)
                for np5 in np5_matches
            )
        )

    def choose_generic_level100_np5(
        self,
        screen: np.ndarray,
        servant_class: str,
    ) -> SupportChoice | None:
        """Choose an unregistered fallback that visibly has level 1xx and NP5."""
        guest_rows = self.guest_support_row_ranges(screen)
        level_matches = [
            match
            for match in self._level100_plus_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        np5_matches = [
            match
            for match in self._np5_matches(screen)
            if not self._point_in_guest_rows(match.center, guest_rows)
        ]
        min_delta, max_delta = self.settings.get(
            "generic_row_marker_delta_y",
            [80, 190],
        )
        pairs = [
            (level.score + np5.score, level, np5)
            for level in level_matches
            for np5 in np5_matches
            if float(min_delta)
            <= np5.center[1] - level.center[1]
            <= float(max_delta)
        ]
        if not pairs:
            return None
        _, _, np5_match = max(
            pairs,
            key=lambda item: (item[0], -item[1].center[1]),
        )
        click_y = np5_match.center[1] + int(
            self.settings.get("generic_row_click_y_offset", -25)
        )
        click_x = int(self.settings.get("generic_row_click_x", 700))
        visual = VisualMatch(  # Keep selection centered on the matching row.
            score=np5_match.score,
            x=click_x - 1,
            y=click_y - 1,
            width=2,
            height=2,
        )
        candidate = {
            "id": f"generic_{servant_class}_lv100_np5_fallback",
            "name": f"100级以上、宝具5 {servant_class}",
            "class": servant_class,
            "level": 100,
            "level_at_least": 100,
            "np_level": 5,
            "np_target": "unknown",
            "np_damage": True,
            "np_color": "",
            "party_slot": int(
                self.settings.get("generic_party_slot", 3)
            ),
            "generic": True,
            "relaxed_support": True,
        }
        return SupportChoice(
            candidate=candidate,
            match=visual,
            enemy_class="berserker_fallback",
            counter_class=servant_class,
        )

    def is_forced_only_screen(self, screen: np.ndarray) -> bool:
        raw_region = self.settings.get(
            "forced_support_empty_region",
            [250, 500, 1450, 850],
        )
        x1 = round(float(raw_region[0]) * screen.shape[1] / self.base_width)
        y1 = round(float(raw_region[1]) * screen.shape[0] / self.base_height)
        x2 = round(float(raw_region[2]) * screen.shape[1] / self.base_width)
        y2 = round(float(raw_region[3]) * screen.shape[0] / self.base_height)
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bright = (hsv[:, :, 2] >= 150) & (hsv[:, :, 1] <= 100)
        bright_ratio = float(np.count_nonzero(bright)) / float(bright.size)
        return (
            edge_ratio
            <= float(
                self.settings.get("forced_support_max_edge_ratio", 0.02)
            )
            and bright_ratio
            <= float(
                self.settings.get("forced_support_max_bright_ratio", 0.12)
            )
        )

    def normal_support_row_ranges(
        self,
        screen: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Locate visible ordinary-player rows by their yellow Support strip."""
        raw_region = self.settings.get(
            "normal_support_marker_roi",
            [1480, 180, 1520, 890],
        )
        x1 = round(float(raw_region[0]) * screen.shape[1] / self.base_width)
        y1 = round(float(raw_region[1]) * screen.shape[0] / self.base_height)
        x2 = round(float(raw_region[2]) * screen.shape[1] / self.base_width)
        y2 = round(float(raw_region[3]) * screen.shape[0] / self.base_height)
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array(
            self.settings.get(
                "normal_support_marker_hsv_lower",
                [15, 80, 100],
            ),
            dtype=np.uint8,
        )
        upper = np.array(
            self.settings.get(
                "normal_support_marker_hsv_upper",
                [45, 255, 255],
            ),
            dtype=np.uint8,
        )
        marker = cv2.inRange(hsv, lower, upper)
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        minimum_pixels = max(
            1,
            round(
                float(
                    self.settings.get(
                        "normal_support_marker_min_row_pixels",
                        3,
                    )
                )
                * scale_x
            ),
        )
        active_rows = np.flatnonzero(
            np.count_nonzero(marker, axis=1) >= minimum_pixels
        )
        if active_rows.size == 0:
            return []
        maximum_gap = max(
            1,
            round(
                float(
                    self.settings.get(
                        "normal_support_marker_max_row_gap",
                        3,
                    )
                )
                * scale_y
            ),
        )
        minimum_height = max(
            1,
            round(
                float(
                    self.settings.get(
                        "normal_support_marker_min_row_height",
                        80,
                    )
                )
                * scale_y
            ),
        )
        ranges: list[tuple[int, int]] = []
        start = previous = int(active_rows[0])
        for raw_y in active_rows[1:]:
            current = int(raw_y)
            if current - previous > maximum_gap:
                if previous - start + 1 >= minimum_height:
                    ranges.append((y1 + start, y1 + previous))
                start = current
            previous = current
        if previous - start + 1 >= minimum_height:
            ranges.append((y1 + start, y1 + previous))
        return ranges

    def has_normal_support_row(self, screen: np.ndarray) -> bool:
        """Detect the yellow Support strip shown on ordinary player rows."""
        if self.normal_support_row_ranges(screen):
            return True
        raw_region = self.settings.get(
            "normal_support_marker_roi",
            [1480, 180, 1520, 890],
        )
        x1 = round(float(raw_region[0]) * screen.shape[1] / self.base_width)
        y1 = round(float(raw_region[1]) * screen.shape[0] / self.base_height)
        x2 = round(float(raw_region[2]) * screen.shape[1] / self.base_width)
        y2 = round(float(raw_region[3]) * screen.shape[0] / self.base_height)
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array(
            self.settings.get(
                "normal_support_marker_hsv_lower",
                [15, 80, 100],
            ),
            dtype=np.uint8,
        )
        upper = np.array(
            self.settings.get(
                "normal_support_marker_hsv_upper",
                [45, 255, 255],
            ),
            dtype=np.uint8,
        )
        marker = cv2.inRange(hsv, lower, upper)
        ratio = float(np.count_nonzero(marker)) / float(marker.size)
        return ratio >= float(
            self.settings.get("normal_support_marker_min_ratio", 0.02)
        )


class BattlePlanner:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.settings = config.get("battle", {})
        self.base_width = int(config["screen"]["base_width"])
        self.base_height = int(config["screen"]["base_height"])
        self.enemy_class_templates: dict[str, np.ndarray] = {}
        # The six-enemy battle HUD uses two compact rows.  Once that layout
        # is positively identified, keep it for the rest of the battle so
        # enemy deaths cannot make the detector fall back to the three-slot
        # coordinates and click an empty position.
        self._enemy_layout_hint: str | None = None
        self.card_disabled_text_template: np.ndarray | None = None
        self.card_action_disabled_text_template: np.ndarray | None = None
        self.card_sleep_status_template: np.ndarray | None = None
        self.np_card_error_text_template: np.ndarray | None = None
        self.np_card_sealed_text_template: np.ndarray | None = None
        if config_path is not None:
            for class_name, value in self.settings.get(
                "enemy_class_templates",
                {},
            ).items():
                path = resolve_from_config(config_path, value)
                if path.is_file():
                    self.enemy_class_templates[str(class_name)] = read_image(
                        path
                    )
            disabled_text_path = self.settings.get(
                "card_disabled_text_template"
            )
            if disabled_text_path:
                path = resolve_from_config(config_path, disabled_text_path)
                if path.is_file():
                    self.card_disabled_text_template = read_image(path)
            action_disabled_text_path = self.settings.get(
                "card_action_disabled_text_template"
            )
            if action_disabled_text_path:
                path = resolve_from_config(
                    config_path,
                    action_disabled_text_path,
                )
                if path.is_file():
                    self.card_action_disabled_text_template = read_image(path)
            sleep_path = self.settings.get("card_sleep_status_template")
            if sleep_path:
                path = resolve_from_config(config_path, sleep_path)
                if path.is_file():
                    self.card_sleep_status_template = read_image(path)
            error_text_path = self.settings.get(
                "np_card_error_text_template"
            )
            if error_text_path:
                path = resolve_from_config(config_path, error_text_path)
                if path.is_file():
                    self.np_card_error_text_template = read_image(path)
            sealed_text_path = self.settings.get(
                "np_card_sealed_text_template"
            )
            if sealed_text_path:
                path = resolve_from_config(config_path, sealed_text_path)
                if path.is_file():
                    self.np_card_sealed_text_template = read_image(path)

    def _scaled_point(
        self,
        point: list[float] | tuple[float, float],
        screen: np.ndarray,
    ) -> tuple[int, int]:
        height, width = screen.shape[:2]
        return (
            round(float(point[0]) * width / self.base_width),
            round(float(point[1]) * height / self.base_height),
        )

    def _scaled_region(
        self,
        region: list[float],
        screen: np.ndarray,
    ) -> tuple[int, int, int, int]:
        x1, y1 = self._scaled_point(region[:2], screen)
        x2, y2 = self._scaled_point(region[2:], screen)
        return x1, y1, x2, y2

    def reset_enemy_layout(self) -> None:
        """Forget the battle-local enemy HUD layout."""
        self._enemy_layout_hint = None

    def _enemy_health_scores_from_regions(
        self,
        screen: np.ndarray,
        regions: dict[Any, Any],
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        for raw_slot, raw_region in regions.items():
            try:
                slot = int(raw_slot)
            except (TypeError, ValueError):
                continue
            if not isinstance(raw_region, (list, tuple)) or len(raw_region) != 4:
                continue
            x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red = (
                ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
                & (hsv[:, :, 1] >= 100)
                & (hsv[:, :, 2] >= 70)
            )
            purple = (
                (hsv[:, :, 0] >= 125)
                & (hsv[:, :, 0] <= 169)
                & (hsv[:, :, 1] >= 75)
                & (hsv[:, :, 2] >= 65)
            )
            yellow_green = (
                (hsv[:, :, 0] >= 18)
                & (hsv[:, :, 0] <= 85)
                & (hsv[:, :, 1] >= 75)
                & (hsv[:, :, 2] >= 65)
            )
            health = red | purple | yellow_green
            scores[slot] = (
                float(np.count_nonzero(health)) / float(health.size)
            )
        return scores

    def _six_enemy_layout_visible(self, screen: np.ndarray) -> bool:
        if self._enemy_layout_hint == "six":
            return True
        regions = self.settings.get("enemy_hp_regions_six", {})
        if not isinstance(regions, dict) or not regions:
            return False
        scores = self._enemy_health_scores_from_regions(screen, regions)
        threshold = float(
            self.settings.get("enemy_six_layout_detect_min_ratio", 0.25)
        )
        upper_slots = tuple(
            int(value)
            for value in self.settings.get(
                "enemy_six_layout_upper_slots",
                [1, 2, 3],
            )
        )
        lower_slots = tuple(
            int(value)
            for value in self.settings.get(
                "enemy_six_layout_lower_slots",
                [4, 5, 6],
            )
        )
        upper_count = sum(scores.get(slot, 0.0) >= threshold for slot in upper_slots)
        lower_count = sum(scores.get(slot, 0.0) >= threshold for slot in lower_slots)
        minimum_per_row = max(
            1,
            int(self.settings.get("enemy_six_layout_min_per_row", 2)),
        )
        if upper_count >= minimum_per_row and lower_count >= minimum_per_row:
            self._enemy_layout_hint = "six"
            return True
        return False

    def _enemy_regions(
        self,
        screen: np.ndarray,
        standard_key: str,
    ) -> dict[Any, Any]:
        if self._six_enemy_layout_visible(screen):
            six = self.settings.get(f"{standard_key}_six", {})
            if isinstance(six, dict) and six:
                return six
        standard = self.settings.get(standard_key, {})
        return standard if isinstance(standard, dict) else {}

    def _enemy_target_point(
        self,
        screen: np.ndarray,
        slot: int,
    ) -> tuple[int, int] | None:
        points = self._enemy_regions(screen, "enemy_target_points")
        raw_point = points.get(str(slot)) or points.get(slot)
        if raw_point is None:
            return None
        return self._scaled_point(raw_point, screen)

    def _np_gauge_metrics(
        self,
        screen: np.ndarray,
        slot: int,
    ) -> tuple[float, float] | None:
        regions = self.settings.get("np_gauge_regions", {})
        raw_region = regions.get(str(slot)) or regions.get(slot)
        if not raw_region:
            return None
        x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        b, g, r = cv2.split(roi)
        gold = (
            (r.astype(np.int16) > 125)
            & (g.astype(np.int16) > 55)
            & (r.astype(np.int16) > g.astype(np.int16) * 1.12)
            & (g.astype(np.int16) > b.astype(np.int16) * 1.15)
        )
        ratio = float(np.count_nonzero(gold)) / float(gold.size)
        column_ratio = gold.mean(axis=0)
        filled_columns = np.flatnonzero(
            column_ratio
            >= float(
                self.settings.get(
                    "np_ready_column_gold_ratio",
                    0.25,
                )
            )
        )
        if filled_columns.size == 0:
            return ratio, 0.0
        filled_extent = float(filled_columns[-1] + 1) / float(
            gold.shape[1]
        )
        return ratio, filled_extent

    def np_ready(self, screen: np.ndarray, slot: int) -> bool:
        metrics = self._np_gauge_metrics(screen, slot)
        if metrics is None:
            return False
        ratio, filled_extent = metrics
        if ratio >= float(self.settings.get("np_ready_gold_ratio", 0.32)):
            return True
        return (
            ratio
            >= float(
                self.settings.get(
                    "np_ready_min_gold_ratio",
                    0.30,
                )
            )
            and filled_extent
            >= float(
                self.settings.get(
                    "np_ready_fill_extent_ratio",
                    0.94,
                )
            )
        )

    def battle_turn_number(self, screen: np.ndarray) -> int | None:
        """Read the real battle turn shown before the ``回合`` label."""
        raw_region = self.settings.get(
            "battle_turn_region",
            [1080, 95, 1152, 138],
        )
        if not isinstance(raw_region, list) or len(raw_region) != 4:
            return None
        x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.where(
            (
                hsv[:, :, 1]
                <= int(
                    self.settings.get(
                        "battle_turn_max_saturation",
                        100,
                    )
                )
            )
            & (
                hsv[:, :, 2]
                >= int(
                    self.settings.get(
                        "battle_turn_min_value",
                        170,
                    )
                )
            ),
            255,
            0,
        ).astype(np.uint8)

        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        min_area = max(6, round(mask.size * 0.002))
        min_height = max(5, round(mask.shape[0] * 0.35))
        max_height = max(min_height, round(mask.shape[0] * 0.85))
        max_width = max(8, round(mask.shape[1] * 0.45))
        components = [
            tuple(int(value) for value in component)
            for component in stats[1:]
            if int(component[4]) >= min_area
            and min_height <= int(component[3]) <= max_height
            and int(component[2]) <= max_width
        ]
        if not 1 <= len(components) <= 2:
            return None

        digits: list[int] = []
        min_confidence = float(
            self.settings.get(
                "battle_turn_min_digit_confidence",
                0.72,
            )
        )
        for x, y, width, height, _ in sorted(components):
            digit, confidence = self._recognize_np_digit(
                mask[y : y + height, x : x + width]
            )
            if digit < 0 or confidence < min_confidence:
                return None
            digits.append(digit)

        value = int("".join(str(digit) for digit in digits))
        return value if 1 <= value <= 99 else None

    def np_gauge_percent(
        self,
        screen: np.ndarray,
        slot: int,
    ) -> int | None:
        """Approximate NP percentage from the bar as an OCR fallback."""
        metrics = self._np_gauge_metrics(screen, slot)
        if metrics is None:
            return None
        _, filled_extent = metrics
        full_extent = float(
            self.settings.get("np_gauge_full_extent_ratio", 0.97)
        )
        if full_extent <= 0:
            return None
        return round(max(0.0, min(1.0, filled_extent / full_extent)) * 100)

    def _recognize_np_digit(
        self,
        mask: np.ndarray,
    ) -> tuple[int, float]:
        glyph = _normalize_digit(mask)
        glyph_on = glyph > 0
        glyph_count = int(np.count_nonzero(glyph_on))
        best_digit = -1
        best_score = 0.0
        for digit, template in _digit_templates():
            template_on = template > 0
            denominator = glyph_count + int(np.count_nonzero(template_on))
            if denominator == 0:
                continue
            score = (
                2.0
                * float(np.count_nonzero(glyph_on & template_on))
                / float(denominator)
            )
            if score > best_score:
                best_digit = digit
                best_score = score
        return best_digit, best_score

    def np_percent(self, screen: np.ndarray, slot: int) -> int | None:
        """Read the displayed numeric NP percentage, including 100% exactly."""
        regions = self.settings.get("np_value_regions", {})
        raw_region = regions.get(str(slot)) or regions.get(slot)
        if not raw_region:
            return None
        x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.where(
            (
                hsv[:, :, 1]
                <= int(self.settings.get("np_value_max_saturation", 100))
            )
            & (
                hsv[:, :, 2]
                >= int(self.settings.get("np_value_min_value", 175))
            ),
            255,
            0,
        ).astype(np.uint8)

        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        min_area = max(6, round(mask.size * 0.0012))
        components = [
            tuple(int(value) for value in component)
            for component in stats[1:]
            if int(component[4]) >= min_area
            and int(component[3]) >= max(5, round(mask.shape[0] * 0.18))
        ]

        percent_x: int | None = None
        for x, y, width, height, _ in components:
            if not (
                mask.shape[0] * 0.15 <= height <= mask.shape[0] * 0.35
                and mask.shape[0] * 0.25 <= y <= mask.shape[0] * 0.65
            ):
                continue
            slash_found = any(
                0 <= slash_x - x <= width + 1
                and abs(slash_y - y) <= 2
                and slash_height >= mask.shape[0] * 0.40
                for (
                    slash_x,
                    slash_y,
                    _,
                    slash_height,
                    _,
                ) in components
            )
            if slash_found:
                percent_x = x
                break
        if percent_x is None:
            return None

        digit_mask = mask[:, :percent_x]
        _, _, digit_stats, _ = cv2.connectedComponentsWithStats(
            digit_mask,
            8,
        )
        digit_parts = [
            tuple(int(value) for value in component)
            for component in digit_stats[1:]
            if int(component[4]) >= min_area
            and int(component[1]) >= round(mask.shape[0] * 0.25)
            and int(component[3]) >= max(5, round(mask.shape[0] * 0.18))
        ]
        groups: list[list[tuple[int, int, int, int, int]]] = []
        for component in sorted(digit_parts):
            x, _, width, _, _ = component
            for group in groups:
                group_x1 = min(item[0] for item in group)
                group_x2 = max(item[0] + item[2] for item in group)
                if x <= group_x2 + 1 and x + width >= group_x1 - 1:
                    group.append(component)
                    break
            else:
                groups.append([component])

        digits: list[int] = []
        min_confidence = float(
            self.settings.get("np_value_min_digit_confidence", 0.56)
        )
        for group in groups:
            gx1 = min(item[0] for item in group)
            gy1 = min(item[1] for item in group)
            gx2 = max(item[0] + item[2] for item in group)
            gy2 = max(item[1] + item[3] for item in group)
            if gy2 - gy1 < mask.shape[0] * 0.40:
                continue
            digit, confidence = self._recognize_np_digit(
                digit_mask[gy1:gy2, gx1:gx2]
            )
            if digit < 0 or confidence < min_confidence:
                return None
            digits.append(digit)

        if not 1 <= len(digits) <= 3:
            return None
        value = int("".join(str(digit) for digit in digits))
        return value if 0 <= value <= 300 else None

    def np_charge_percent(
        self,
        screen: np.ndarray,
        slot: int,
    ) -> tuple[int | None, str]:
        """Prefer numeric OCR and keep the gauge only as a safe fallback."""
        value = self.np_percent(screen, slot)
        if value is not None:
            return value, "数值"
        return self.np_gauge_percent(screen, slot), "图形条兜底"

    def np_card_banner_present(self, screen: np.ndarray, slot: int) -> bool:
        """Detect the stable coloured NP-name burst in the card's lower half."""
        points = self.settings.get("card_points", {})
        raw_point = points.get(f"np{slot}")
        if not raw_point:
            return False
        center_x, center_y = self._scaled_point(raw_point, screen)
        offset = self.settings.get(
            "np_card_banner_roi_offset",
            [-105, 0, 105, 155],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, center_x + round(float(offset[0]) * scale_x))
        y1 = max(0, center_y + round(float(offset[1]) * scale_y))
        x2 = min(screen.shape[1], center_x + round(float(offset[2]) * scale_x))
        y2 = min(screen.shape[0], center_y + round(float(offset[3]) * scale_y))
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        red = (
            ((hue <= 14) | (hue >= 166))
            & (saturation >= 90)
            & (value >= 75)
        )
        green = (
            (hue >= 28)
            & (hue <= 88)
            & (saturation >= 90)
            & (value >= 70)
        )
        blue = (
            (hue >= 90)
            & (hue <= 140)
            & (saturation >= 80)
            & (value >= 70)
        )
        colour_mask = red | green | blue
        colour_ratio = float(np.count_nonzero(colour_mask)) / float(
            colour_mask.size
        )
        component_mask = np.where(colour_mask, 255, 0).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            component_mask,
            8,
        )
        component_area_ratio = 0.0
        component_width_ratio = 0.0
        if count > 1:
            largest_index = 1 + int(
                np.argmax(stats[1:, cv2.CC_STAT_AREA])
            )
            component_area_ratio = float(
                stats[largest_index, cv2.CC_STAT_AREA]
            ) / float(component_mask.size)
            component_width_ratio = float(
                stats[largest_index, cv2.CC_STAT_WIDTH]
            ) / float(component_mask.shape[1])

        text_mask = (
            (saturation <= 95)
            & (value >= 155)
        )
        text_ratio = float(np.count_nonzero(text_mask)) / float(
            text_mask.size
        )
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edge_ratio = float(
            np.count_nonzero(cv2.Canny(gray, 60, 140))
        ) / float(gray.size)
        return (
            colour_ratio
            >= float(
                self.settings.get("np_card_banner_min_color_ratio", 0.10)
            )
            and colour_ratio
            <= float(
                self.settings.get("np_card_banner_max_color_ratio", 0.48)
            )
            and component_area_ratio
            >= float(
                self.settings.get(
                    "np_card_banner_min_component_area_ratio",
                    0.05,
                )
            )
            and component_area_ratio
            <= float(
                self.settings.get(
                    "np_card_banner_max_component_area_ratio",
                    0.35,
                )
            )
            and component_width_ratio
            >= float(
                self.settings.get(
                    "np_card_banner_min_component_width_ratio",
                    0.30,
                )
            )
            and component_width_ratio
            <= float(
                self.settings.get(
                    "np_card_banner_max_component_width_ratio",
                    0.90,
                )
            )
            and text_ratio
            >= float(
                self.settings.get("np_card_banner_min_text_ratio", 0.08)
            )
            and edge_ratio
            >= float(
                self.settings.get("np_card_banner_min_edge_ratio", 0.14)
            )
        )

    def np_card_selectable(self, screen: np.ndarray, slot: int) -> bool:
        """Detect an actually visible, bright NP card on the command screen."""
        points = self.settings.get("card_points", {})
        raw_point = points.get(f"np{slot}")
        if not raw_point:
            return False
        disabled, _ = self.np_card_disabled(screen, slot)
        if disabled:
            return False
        center_x, center_y = self._scaled_point(raw_point, screen)
        offset = self.settings.get(
            "np_card_selectable_roi_offset",
            [-100, -100, 100, 150],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, center_x + round(float(offset[0]) * scale_x))
        y1 = max(0, center_y + round(float(offset[1]) * scale_y))
        x2 = min(
            screen.shape[1],
            center_x + round(float(offset[2]) * scale_x),
        )
        y2 = min(
            screen.shape[0],
            center_y + round(float(offset[3]) * scale_y),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        bright_ratio = float(
            np.count_nonzero(
                value
                >= int(
                    self.settings.get(
                        "np_card_selectable_bright_value",
                        160,
                    )
                )
            )
        ) / float(value.size)
        dark_ratio = float(
            np.count_nonzero(
                value
                <= int(
                    self.settings.get(
                        "np_card_selectable_dark_value",
                        90,
                    )
                )
            )
        ) / float(value.size)
        saturated_ratio = float(
            np.count_nonzero(
                hsv[:, :, 1]
                >= int(
                    self.settings.get(
                        "np_card_selectable_saturated_value",
                        100,
                    )
                )
            )
        ) / float(value.size)
        colored_bright_ratio = float(
            np.count_nonzero(
                (
                    value
                    >= int(
                        self.settings.get(
                            "np_card_selectable_dark_art_color_value",
                            140,
                        )
                    )
                )
                & (
                    hsv[:, :, 1]
                    >= int(
                        self.settings.get(
                            "np_card_selectable_dark_art_color_saturation",
                            100,
                        )
                    )
                )
            )
        ) / float(value.size)
        white_bright_ratio = float(
            np.count_nonzero(
                (
                    value
                    >= int(
                        self.settings.get(
                            "np_card_selectable_bright_value",
                            160,
                        )
                    )
                )
                & (
                    hsv[:, :, 1]
                    < int(
                        self.settings.get(
                            "np_card_selectable_dark_art_white_saturation",
                            80,
                        )
                    )
                )
            )
        ) / float(value.size)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edge_ratio = float(
            np.count_nonzero(cv2.Canny(gray, 60, 140))
        ) / float(gray.size)
        regular_bright_card = (
            bright_ratio
            >= float(
                self.settings.get(
                    "np_card_selectable_min_bright_ratio",
                    0.42,
                )
            )
            and dark_ratio
            <= float(
                self.settings.get(
                    "np_card_selectable_max_dark_ratio",
                    0.42,
                )
            )
            and saturated_ratio
            <= float(
                self.settings.get(
                    "np_card_selectable_max_saturated_ratio",
                    0.65,
                )
            )
            and edge_ratio
            >= float(
                self.settings.get(
                    "np_card_selectable_min_edge_ratio",
                    0.16,
                )
            )
        )
        # Some fully usable NPs have unusually dark character artwork.  A
        # brightness-only gate rejected those cards even when the numeric NP
        # value was already over 100%.  Keep the original conservative path,
        # then admit a narrowly-defined dark-art card: it must still contain
        # coloured card detail and edges, and must not be dominated by the
        # large white overlay used by sealed/unavailable NP cards.
        dark_art_card = (
            bright_ratio
            >= float(
                self.settings.get(
                    "np_card_selectable_dark_art_min_bright_ratio",
                    0.25,
                )
            )
            and dark_ratio
            <= float(
                self.settings.get(
                    "np_card_selectable_dark_art_max_dark_ratio",
                    0.475,
                )
            )
            and saturated_ratio
            <= float(
                self.settings.get(
                    "np_card_selectable_dark_art_max_saturated_ratio",
                    0.50,
                )
            )
            and colored_bright_ratio
            >= float(
                self.settings.get(
                    "np_card_selectable_dark_art_min_colored_ratio",
                    0.18,
                )
            )
            and white_bright_ratio
            <= float(
                self.settings.get(
                    "np_card_selectable_dark_art_max_white_ratio",
                    0.15,
                )
            )
            and edge_ratio
            >= float(
                self.settings.get(
                    "np_card_selectable_dark_art_min_edge_ratio",
                    0.16,
                )
            )
        )
        # A few usable NPs have a dark or background-coloured portrait, while
        # the lower red/green/blue NP-name burst remains stable.  Admit that
        # narrow case without broadly relaxing the full-card thresholds.
        # The explicit action-disable/sleep/ERROR/seal detector above still
        # has absolute priority, and the conservative full-card darkness
        # limits keep old dimmed/sealed cards out of this rescue path.
        lower_banner_card = (
            self.np_card_banner_present(screen, slot)
            and dark_ratio
            <= float(
                self.settings.get(
                    "np_card_banner_rescue_max_dark_ratio",
                    0.46,
                )
            )
            and saturated_ratio
            <= float(
                self.settings.get(
                    "np_card_banner_rescue_max_saturated_ratio",
                    0.22,
                )
            )
        )
        return regular_bright_card or dark_art_card or lower_banner_card

    def skill_ready(
        self,
        screen: np.ndarray,
        point: list[float] | tuple[float, float],
    ) -> bool:
        center_x, center_y = self._scaled_point(point, screen)
        offset = self.settings.get(
            "skill_ready_roi_offset",
            [-35, -40, 35, 40],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, center_x + round(float(offset[0]) * scale_x))
        y1 = max(0, center_y + round(float(offset[1]) * scale_y))
        x2 = min(
            screen.shape[1],
            center_x + round(float(offset[2]) * scale_x),
        )
        y2 = min(
            screen.shape[0],
            center_y + round(float(offset[3]) * scale_y),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        bright = hsv[:, :, 2] >= int(
            self.settings.get("skill_ready_min_value", 140)
        )
        ratio = float(np.count_nonzero(bright)) / float(bright.size)
        if ratio >= float(
            self.settings.get("skill_ready_bright_ratio", 0.55)
        ):
            return True

        # Some usable skills have naturally dark red/black artwork.  Their
        # full-button brightness can resemble cooldown, while the upper half
        # remains much brighter than a genuinely dimmed cooldown icon.  Keep
        # the legacy whole-button check and add this cooldown-text-free probe.
        top_fraction = min(
            1.0,
            max(
                0.1,
                float(
                    self.settings.get(
                        "skill_ready_dark_icon_top_fraction",
                        0.5,
                    )
                ),
            ),
        )
        top_height = max(1, round(bright.shape[0] * top_fraction))
        top_ratio = float(
            np.count_nonzero(bright[:top_height])
        ) / float(bright[:top_height].size)
        return top_ratio >= float(
            self.settings.get(
                "skill_ready_dark_icon_min_bright_ratio",
                0.22,
            )
        )

    def skill_target_points(
        self,
        screen: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Detect the re-centered 1/2/3-servant skill-target layout."""
        raw_layouts = self.settings.get(
            "skill_target_layout_points",
            {
                "1": [[800, 470]],
                "2": [[610, 470], [990, 470]],
                "3": [[420, 470], [800, 470], [1180, 470]],
            },
        )
        offset = self.settings.get(
            "skill_target_portrait_roi_offset",
            [-80, -140, 80, 160],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        min_bright_value = int(
            self.settings.get("skill_target_min_bright_value", 100)
        )
        min_bright_ratio = float(
            self.settings.get("skill_target_min_bright_ratio", 0.10)
        )
        min_edge_ratio = float(
            self.settings.get("skill_target_min_edge_ratio", 0.045)
        )

        def portrait_visible(raw_point: list[float]) -> bool:
            center_x, center_y = self._scaled_point(raw_point, screen)
            x1 = max(0, center_x + round(float(offset[0]) * scale_x))
            y1 = max(0, center_y + round(float(offset[1]) * scale_y))
            x2 = min(
                screen.shape[1],
                center_x + round(float(offset[2]) * scale_x),
            )
            y2 = min(
                screen.shape[0],
                center_y + round(float(offset[3]) * scale_y),
            )
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                return False
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            bright_ratio = float(
                np.count_nonzero(
                    hsv[:, :, 2] >= min_bright_value
                )
            ) / float(hsv.shape[0] * hsv.shape[1])
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edge_ratio = float(
                np.count_nonzero(cv2.Canny(gray, 60, 140))
            ) / float(gray.size)
            return (
                bright_ratio >= min_bright_ratio
                and edge_ratio >= min_edge_ratio
            )

        parsed_layouts: list[tuple[int, list[list[float]]]] = []
        for raw_count, raw_points in raw_layouts.items():
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count <= 0 or len(raw_points) != count:
                continue
            parsed_layouts.append((count, list(raw_points)))

        # Check the largest layout first.  A three-target screen also has a
        # portrait in the center, but its two side portraits distinguish it
        # from a genuinely single-target screen.
        for _count, raw_points in sorted(
            parsed_layouts,
            key=lambda item: item[0],
            reverse=True,
        ):
            if all(portrait_visible(raw_point) for raw_point in raw_points):
                return [
                    self._scaled_point(raw_point, screen)
                    for raw_point in raw_points
                ]
        return []

    def skill_target_point(
        self,
        screen: np.ndarray,
        target_slot: int,
        active_slots: list[int],
    ) -> tuple[int, int] | None:
        """Map a current battle slot to its re-centered target portrait."""
        points = self.skill_target_points(screen)
        if len(points) == 1:
            return points[0]
        ordered_slots = sorted({int(slot) for slot in active_slots})
        if (
            len(points) != len(ordered_slots)
            or target_slot not in ordered_slots
        ):
            return None
        return points[ordered_slots.index(target_slot)]

    def single_skill_target_point(
        self,
        screen: np.ndarray,
    ) -> tuple[int, int] | None:
        """Return the centered portrait when only one skill target is offered."""
        points = self.skill_target_points(screen)
        return points[0] if len(points) == 1 else None

    def skill_confirmation_visible(self, screen: np.ndarray) -> bool:
        """Detect the optional “技能使用 / 决定” confirmation overlay."""
        def region_metrics(
            raw_region: list[float],
        ) -> tuple[float, float]:
            x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0, 0.0
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white = (
                (hsv[:, :, 1] <= int(
                    self.settings.get(
                        "skill_confirm_white_max_saturation",
                        45,
                    )
                ))
                & (hsv[:, :, 2] >= int(
                    self.settings.get(
                        "skill_confirm_white_min_value",
                        170,
                    )
                ))
            )
            dark = hsv[:, :, 2] <= int(
                self.settings.get("skill_confirm_dark_max_value", 80)
            )
            return (
                float(np.count_nonzero(white)) / float(white.size),
                float(np.count_nonzero(dark)) / float(dark.size),
            )

        confirm_white, _ = region_metrics(
            self.settings.get(
                "skill_confirm_button_region",
                [825, 480, 1275, 575],
            )
        )
        cancel_white, _ = region_metrics(
            self.settings.get(
                "skill_cancel_button_region",
                [325, 480, 735, 575],
            )
        )
        _, panel_dark = region_metrics(
            self.settings.get(
                "skill_confirm_panel_region",
                [170, 180, 1430, 600],
            )
        )
        _, title_dark = region_metrics(
            self.settings.get(
                "skill_confirm_title_region",
                [600, 185, 1000, 270],
            )
        )
        return (
            confirm_white
            >= float(
                self.settings.get(
                    "skill_confirm_button_min_white_ratio",
                    0.55,
                )
            )
            and cancel_white
            >= float(
                self.settings.get(
                    "skill_cancel_button_min_white_ratio",
                    0.55,
                )
            )
            and panel_dark
            >= float(
                self.settings.get(
                    "skill_confirm_panel_min_dark_ratio",
                    0.60,
                )
            )
            and title_dark
            >= float(
                self.settings.get(
                    "skill_confirm_title_min_dark_ratio",
                    0.70,
                )
            )
        )

    def party_health_ratios(
        self,
        screen: np.ndarray,
    ) -> dict[int, float]:
        """Estimate each visible servant's HP bar fill from its coloured run.

        The servant HP bar is blue normally and turns red at critical health.
        Treating only blue pixels as HP made a nearly defeated servant look
        absent.  Conversely, using the class icon as an independent presence
        signal allowed a defeated servant's fading icon to look alive.
        """
        regions = self.settings.get(
            "party_hp_fill_regions",
            {
                "1": [190, 800, 390, 814],
                "2": [590, 800, 790, 814],
                "3": [990, 800, 1190, 814],
            },
        )
        full_width = float(
            self.settings.get("party_hp_full_run_width", 170)
        )
        column_ratio = float(
            self.settings.get(
                "party_hp_color_column_ratio",
                self.settings.get("party_hp_blue_column_ratio", 0.60),
            )
        )
        run_widths = self._party_hp_run_widths(
            screen,
            regions,
            column_ratio=column_ratio,
        )
        scale_x = screen.shape[1] / self.base_width
        scaled_full_width = max(1.0, full_width * scale_x)
        return {
            slot: min(1.0, longest / scaled_full_width)
            for slot, longest in run_widths.items()
        }

    def _party_hp_run_widths(
        self,
        screen: np.ndarray,
        regions: dict[str | int, list[int]],
        *,
        column_ratio: float,
    ) -> dict[int, int]:
        """Return the longest stable blue/red HP run for each party slot."""
        output: dict[int, float] = {}
        for slot in (1, 2, 3):
            raw_region = regions.get(str(slot)) or regions.get(slot)
            if not raw_region:
                continue
            x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            blue = (
                (hsv[:, :, 0] >= 80)
                & (hsv[:, :, 0] <= 115)
                & (hsv[:, :, 1] >= 80)
                & (hsv[:, :, 2] >= 80)
            )
            red = (
                (
                    (hsv[:, :, 0] <= 10)
                    | (hsv[:, :, 0] >= 170)
                )
                & (hsv[:, :, 1] >= 90)
                & (hsv[:, :, 2] >= 70)
            )
            # Requiring the colour in most rows rejects diagonal card/effect
            # fragments that merely cross the HP region.
            columns = np.mean(blue | red, axis=0) >= column_ratio
            longest = 0
            current = 0
            for present in columns:
                if present:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            output[slot] = longest
        return output

    def party_identity_features(
        self,
        screen: np.ndarray,
        active_slots: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Build stable visual fingerprints from the visible servant names.

        HP and NP gauges change continuously, so they are deliberately
        excluded.  The name/class line remains stable for the same on-field
        servant and changes when a back-line servant replaces a defeated one.
        """
        regions = self.settings.get(
            "party_identity_regions",
            {
                "1": [135, 845, 395, 900],
                "2": [535, 845, 795, 900],
                "3": [935, 845, 1195, 900],
            },
        )
        requested = set(active_slots or (1, 2, 3))
        feature_width = int(
            self.settings.get("party_identity_feature_width", 96)
        )
        feature_height = int(
            self.settings.get("party_identity_feature_height", 20)
        )
        output: dict[int, np.ndarray] = {}
        for slot in (1, 2, 3):
            if slot not in requested:
                continue
            raw_region = regions.get(str(slot)) or regions.get(slot)
            if not raw_region:
                continue
            x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 55, 140)
            resized = cv2.resize(
                edges,
                (feature_width, feature_height),
                interpolation=cv2.INTER_AREA,
            )
            feature = resized.astype(np.float32).reshape(-1)
            feature -= float(feature.mean())
            norm = float(np.linalg.norm(feature))
            if norm <= 1e-6:
                continue
            output[slot] = feature / norm
        return output

    def active_party_slots(self, screen: np.ndarray) -> list[int]:
        regions = self.settings.get(
            "party_hp_fill_regions",
            self.settings.get("active_hp_regions", {}),
        )
        column_ratio = float(
            self.settings.get(
                "party_hp_color_column_ratio",
                self.settings.get("party_hp_blue_column_ratio", 0.60),
            )
        )
        hp_run_widths = self._party_hp_run_widths(
            screen,
            regions,
            column_ratio=column_ratio,
        )
        ratio_threshold = float(
            self.settings.get("active_hp_blue_ratio", 0.015)
        )
        configured_min_run = self.settings.get("active_hp_min_run_width")
        max_run_ratio = float(
            self.settings.get("active_hp_max_run_ratio", 0.96)
        )
        icon_regions = self.settings.get("active_icon_regions", {})
        icon_threshold = float(
            self.settings.get("active_icon_foreground_ratio", 0.18)
        )
        icon_white_threshold = float(
            self.settings.get("active_icon_white_ratio", 0.08)
        )
        hud_edge_threshold = float(
            self.settings.get("active_hud_edge_ratio", 0.04)
        )
        hud_regions = (
            self.settings.get("np_value_regions", {}),
            self.settings.get("party_identity_regions", {}),
        )
        active: list[int] = []
        for slot in (1, 2, 3):
            raw_region = regions.get(str(slot)) or regions.get(slot)
            has_hp_bar = False
            if raw_region:
                reference_width = max(1, int(raw_region[2]) - int(raw_region[0]))
                reference_min_run = (
                    float(configured_min_run)
                    if configured_min_run is not None
                    else ratio_threshold * reference_width
                )
                scale_x = screen.shape[1] / self.base_width
                min_run = max(1.0, reference_min_run * scale_x)
                max_run = (
                    reference_width
                    * scale_x
                    * max(0.0, min(1.0, max_run_ratio))
                )
                run_width = hp_run_widths.get(slot, 0)
                has_hp_bar = min_run <= run_width <= max_run

            raw_icon_region = (
                icon_regions.get(str(slot)) or icon_regions.get(slot)
            )
            has_class_icon = False
            has_coloured_class_icon = False
            if raw_icon_region:
                x1, y1, x2, y2 = self._scaled_region(
                    raw_icon_region,
                    screen,
                )
                roi = screen[y1:y2, x1:x2]
                if roi.size:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    white = (
                        (hsv[:, :, 1] <= 80)
                        & (hsv[:, :, 2] >= 140)
                    )
                    gold = (
                        (hsv[:, :, 0] >= 8)
                        & (hsv[:, :, 0] <= 45)
                        & (hsv[:, :, 1] >= 70)
                        & (hsv[:, :, 2] >= 80)
                    )
                    foreground = white | gold
                    foreground_ratio = (
                        float(np.count_nonzero(foreground))
                        / float(foreground.size)
                    )
                    white_ratio = (
                        float(np.count_nonzero(white))
                        / float(white.size)
                    )
                    has_coloured_class_icon = (
                        foreground_ratio >= icon_threshold
                    )
                    has_class_icon = (
                        has_coloured_class_icon
                        and white_ratio >= icon_white_threshold
                    )

            # A servant revived by Guts can be left at 1 HP.  At 1600x900 the
            # coloured HP fill then occupies less than one stable column and
            # legitimately disappears from the detector.  In that case use a
            # second, independent signal: the class icon plus the sharp text /
            # frame edges in the NP and servant-name HUD.  Requiring both keeps
            # a plain gold battlefield or a briefly lingering icon from
            # resurrecting an empty slot.
            has_party_hud = False
            if raw_region and not has_hp_bar and has_coloured_class_icon:
                for region_map in hud_regions:
                    raw_hud_region = (
                        region_map.get(str(slot)) or region_map.get(slot)
                    )
                    if not raw_hud_region:
                        continue
                    hx1, hy1, hx2, hy2 = self._scaled_region(
                        raw_hud_region,
                        screen,
                    )
                    hud_roi = screen[hy1:hy2, hx1:hx2]
                    if hud_roi.size == 0:
                        continue
                    hud_gray = cv2.cvtColor(hud_roi, cv2.COLOR_BGR2GRAY)
                    hud_edges = cv2.Canny(hud_gray, 55, 140)
                    edge_ratio = (
                        float(np.count_nonzero(hud_edges))
                        / float(hud_edges.size)
                    )
                    if edge_ratio >= hud_edge_threshold:
                        has_party_hud = True
                        break

            # HP fill remains the primary signal.  Icon-only detection is used
            # only by older/minimal profiles without HP regions; configured
            # profiles require the extra HUD evidence for the 1-HP fallback.
            if (raw_region and (has_hp_bar or has_party_hud)) or (
                not raw_region and has_class_icon
            ):
                active.append(slot)
        return active

    def enemy_health_scores(
        self,
        screen: np.ndarray,
    ) -> dict[int, float]:
        regions = self._enemy_regions(screen, "enemy_hp_regions")
        return self._enemy_health_scores_from_regions(screen, regions)

    def enemy_classes(
        self,
        screen: np.ndarray,
    ) -> dict[int, tuple[str, float]]:
        if not self.enemy_class_templates:
            return {}
        regions = self._enemy_regions(screen, "enemy_class_regions")
        threshold = float(
            self.settings.get("enemy_class_match_threshold", 0.62)
        )
        detected: dict[int, tuple[str, float]] = {}
        for raw_slot, raw_region in regions.items():
            try:
                slot = int(raw_slot)
            except (TypeError, ValueError):
                continue
            if not raw_region:
                continue
            x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            best_name: str | None = None
            best_score = -1.0
            for class_name, template in self.enemy_class_templates.items():
                scaled = _scaled_template(
                    template,
                    screen,
                    self.base_width,
                    self.base_height,
                )
                if (
                    scaled.shape[0] > roi.shape[0]
                    or scaled.shape[1] > roi.shape[1]
                ):
                    continue
                result = cv2.matchTemplate(
                    roi_gray,
                    cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY),
                    cv2.TM_CCOEFF_NORMED,
                )
                _, score, _, _ = cv2.minMaxLoc(result)
                if score > best_score:
                    best_name = class_name
                    best_score = float(score)
            if best_name is not None and best_score >= threshold:
                detected[slot] = best_name, best_score
        return detected

    def class_advantage(
        self,
        attacker_class: str | None,
        defender_class: str | None,
    ) -> bool:
        configured = self.settings.get("class_advantage_map", {})
        if attacker_class in configured:
            targets = configured.get(attacker_class, [])
            return defender_class in {str(value) for value in targets}
        return has_attack_advantage(attacker_class, defender_class)

    def enemy_charge_scores(
        self,
        screen: np.ndarray,
    ) -> dict[int, float]:
        regions = self._enemy_regions(screen, "enemy_charge_regions")
        scores: dict[int, float] = {}
        for raw_slot, raw_region in regions.items():
            try:
                slot = int(raw_slot)
            except (TypeError, ValueError):
                continue
            if not raw_region:
                continue
            x1, y1, x2, y2 = self._scaled_region(raw_region, screen)
            roi = screen[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            yellow = (
                (hsv[:, :, 0] >= 8)
                & (hsv[:, :, 0] <= 45)
                & (hsv[:, :, 1] >= 80)
                & (hsv[:, :, 2] >= 80)
            )
            red = (
                ((hsv[:, :, 0] <= 7) | (hsv[:, :, 0] >= 170))
                & (hsv[:, :, 1] >= 120)
                & (hsv[:, :, 2] >= 100)
            )
            charged = yellow | red
            scores[slot] = (
                float(np.count_nonzero(charged)) / float(charged.size)
            )
        return scores

    def alive_enemy_slots(self, screen: np.ndarray) -> list[int]:
        health_scores = self.enemy_health_scores(screen)
        minimum_health = float(
            self.settings.get("enemy_hp_min_ratio", 0.025)
        )
        return [
            slot
            for slot, health in health_scores.items()
            if health >= minimum_health
        ]

    def priority_enemy(
        self,
        screen: np.ndarray,
        *,
        attacker_class: str | None = None,
        prefer_class_advantage: bool = True,
        prefer_charge: bool = True,
        prefer_high_hp: bool = False,
    ) -> tuple[
        int,
        tuple[int, int],
        float,
        float,
        str | None,
        bool,
    ] | None:
        health_scores = self.enemy_health_scores(screen)
        charge_scores = self.enemy_charge_scores(screen)
        alive = self.alive_enemy_slots(screen)
        if not alive:
            return None
        classes = self.enemy_classes(screen)

        def rank(slot: int) -> tuple[float, float, float]:
            enemy_class = classes.get(slot, (None, 0.0))[0]
            advantage = (
                1.0
                if prefer_class_advantage
                and self.class_advantage(attacker_class, enemy_class)
                else 0.0
            )
            charge = charge_scores.get(slot, 0.0) if prefer_charge else 0.0
            health = health_scores.get(slot, 0.0)
            if prefer_high_hp:
                return advantage, health, charge
            return advantage, charge, health

        slot = max(alive, key=rank)
        target_point = self._enemy_target_point(screen, slot)
        if target_point is None:
            return None
        enemy_class = classes.get(slot, (None, 0.0))[0]
        has_advantage = (
            prefer_class_advantage
            and self.class_advantage(attacker_class, enemy_class)
        )
        return (
            slot,
            target_point,
            charge_scores.get(slot, 0.0),
            health_scores[slot],
            enemy_class,
            has_advantage,
        )

    def highest_hp_enemy(
        self,
        screen: np.ndarray,
    ) -> tuple[int, tuple[int, int], float] | None:
        scores = self.enemy_health_scores(screen)
        if not scores:
            return None
        slot, score = max(scores.items(), key=lambda item: item[1])
        minimum = float(self.settings.get("enemy_hp_min_ratio", 0.025))
        if score < minimum:
            return None
        target_point = self._enemy_target_point(screen, slot)
        if target_point is None:
            return None
        return slot, target_point, score

    def _card_color(
        self,
        screen: np.ndarray,
        point: tuple[int, int],
    ) -> tuple[str, float]:
        offset = self.settings.get("card_color_roi_offset", [-125, -35, 125, 105])
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, point[0] + round(float(offset[0]) * scale_x))
        y1 = max(0, point[1] + round(float(offset[1]) * scale_y))
        x2 = min(
            screen.shape[1],
            point[0] + round(float(offset[2]) * scale_x),
        )
        y2 = min(
            screen.shape[0],
            point[1] + round(float(offset[3]) * scale_y),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown", 0.0
        b, g, r = [channel.astype(np.int16) for channel in cv2.split(roi)]
        bright = np.maximum(np.maximum(b, g), r) > 85
        saturated = np.maximum(np.maximum(b, g), r) - np.minimum(
            np.minimum(b, g), r
        ) > 35
        valid = bright & saturated
        total = max(1, int(np.count_nonzero(valid)))
        masks = {
            "buster": valid & (r > g * 1.18) & (r > b * 1.15),
            "arts": valid & (b > g * 1.10) & (b > r * 1.18),
            "quick": valid & (g > r * 1.12) & (g > b * 1.05),
        }
        scores = {
            name: float(np.count_nonzero(mask)) / total
            for name, mask in masks.items()
        }
        color = max(scores, key=scores.get)
        return color, scores[color]

    def _card_status_template_score(
        self,
        screen: np.ndarray,
        point: tuple[int, int],
        template: np.ndarray | None,
        *,
        white_mask: bool,
        roi_offset_key: str = "card_disabled_roi_offset",
    ) -> float:
        if template is None:
            return 0.0
        offset = self.settings.get(
            roi_offset_key,
            [-140, -140, 140, 0],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, point[0] + round(float(offset[0]) * scale_x))
        y1 = max(0, point[1] + round(float(offset[1]) * scale_y))
        x2 = min(
            screen.shape[1],
            point[0] + round(float(offset[2]) * scale_x),
        )
        y2 = min(
            screen.shape[0],
            point[1] + round(float(offset[3]) * scale_y),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        base_template = _scaled_template(
            template,
            screen,
            self.base_width,
            self.base_height,
        )
        if white_mask:
            lower = np.array(
                self.settings.get(
                    "card_disabled_text_hsv_lower",
                    [0, 0, 120],
                ),
                dtype=np.uint8,
            )
            upper = np.array(
                self.settings.get(
                    "card_disabled_text_hsv_upper",
                    [179, 100, 255],
                ),
                dtype=np.uint8,
            )
            source = cv2.inRange(
                cv2.cvtColor(roi, cv2.COLOR_BGR2HSV),
                lower,
                upper,
            )
        else:
            source = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        best = 0.0
        for raw_scale in self.settings.get(
            "card_disabled_template_scales",
            [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05],
        ):
            template_scale = float(raw_scale)
            width = max(2, round(base_template.shape[1] * template_scale))
            height = max(2, round(base_template.shape[0] * template_scale))
            if roi.shape[1] < width or roi.shape[0] < height:
                continue
            scaled = cv2.resize(
                base_template,
                (width, height),
                interpolation=(
                    cv2.INTER_AREA
                    if template_scale < 1.0
                    else cv2.INTER_CUBIC
                ),
            )
            if white_mask:
                template_source = cv2.inRange(
                    cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV),
                    lower,
                    upper,
                )
            else:
                template_source = cv2.cvtColor(
                    scaled,
                    cv2.COLOR_BGR2GRAY,
                )
            result = cv2.matchTemplate(
                source,
                template_source,
                cv2.TM_CCOEFF_NORMED,
            )
            _, score, _, _ = cv2.minMaxLoc(result)
            best = max(best, float(score))
        return best

    def np_card_disabled(
        self,
        screen: np.ndarray,
        slot: int,
    ) -> tuple[bool, float]:
        """Detect an NP blocked by action-disable, sleep, ERROR, or seal."""
        points = self.settings.get("card_points", {})
        raw_point = points.get(f"np{slot}")
        if not raw_point:
            return False, 0.0
        point = self._scaled_point(raw_point, screen)
        text_score = self._card_status_template_score(
            screen,
            point,
            self.card_disabled_text_template,
            white_mask=True,
            roi_offset_key="np_card_disabled_roi_offset",
        )
        sleep_score = self._card_status_template_score(
            screen,
            point,
            self.card_sleep_status_template,
            white_mask=False,
            roi_offset_key="np_card_disabled_roi_offset",
        )
        error_score = self._card_status_template_score(
            screen,
            point,
            self.np_card_error_text_template,
            white_mask=True,
            roi_offset_key="np_card_disabled_roi_offset",
        )
        sealed_score = self._card_status_template_score(
            screen,
            point,
            self.np_card_sealed_text_template,
            white_mask=True,
            roi_offset_key="np_card_disabled_roi_offset",
        )
        disabled = (
            text_score
            >= float(
                self.settings.get(
                    "np_card_disabled_text_threshold",
                    0.60,
                )
            )
            or sleep_score
            >= float(
                self.settings.get(
                    "np_card_sleep_status_threshold",
                    0.68,
                )
            )
            or error_score
            >= float(
                self.settings.get(
                    "np_card_error_text_threshold",
                    0.80,
                )
            )
            or sealed_score
            >= float(
                self.settings.get(
                    "np_card_sealed_text_threshold",
                    0.78,
                )
            )
        )
        return disabled, max(
            text_score,
            sleep_score,
            error_score,
            sealed_score,
        )

    def _card_disabled(
        self,
        screen: np.ndarray,
        point: tuple[int, int],
    ) -> tuple[bool, float]:
        text_score = self._card_status_template_score(
            screen,
            point,
            self.card_disabled_text_template,
            white_mask=True,
        )
        action_disabled_score = self._card_status_template_score(
            screen,
            point,
            self.card_action_disabled_text_template,
            white_mask=True,
        )
        sleep_score = self._card_status_template_score(
            screen,
            point,
            self.card_sleep_status_template,
            white_mask=False,
        )
        disabled = (
            text_score
            >= float(
                self.settings.get(
                    "card_disabled_text_threshold",
                    0.75,
                )
            )
            or sleep_score
            >= float(
                self.settings.get(
                    "card_sleep_status_threshold",
                    0.68,
                )
            )
            or action_disabled_score
            >= float(
                self.settings.get(
                    "card_action_disabled_text_threshold",
                    0.80,
                )
            )
        )
        return disabled, max(
            text_score,
            sleep_score,
            action_disabled_score,
        )

    def _owner_feature(
        self,
        screen: np.ndarray,
        point: tuple[int, int],
    ) -> np.ndarray:
        offset = self.settings.get(
            "card_portrait_roi_offset",
            [-72, -165, 72, -35],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, point[0] + round(float(offset[0]) * scale_x))
        y1 = max(0, point[1] + round(float(offset[1]) * scale_y))
        x2 = min(
            screen.shape[1],
            point[0] + round(float(offset[2]) * scale_x),
        )
        y2 = min(
            screen.shape[0],
            point[1] + round(float(offset[3]) * scale_y),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(256, dtype=np.float32)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        feature = gray.astype(np.float32).reshape(-1)
        feature -= float(feature.mean())
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 1e-6 else feature

    def _card_support_marker(
        self,
        screen: np.ndarray,
        point: tuple[int, int],
    ) -> tuple[bool, float]:
        """Detect the white horizontal '+助战' badge on a command card."""
        offset = self.settings.get(
            "support_card_marker_roi_offset",
            [0, -180, 145, -125],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, point[0] + round(float(offset[0]) * scale_x))
        y1 = max(0, point[1] + round(float(offset[1]) * scale_y))
        x2 = min(
            screen.shape[1],
            point[0] + round(float(offset[2]) * scale_x),
        )
        y2 = min(
            screen.shape[0],
            point[1] + round(float(offset[3]) * scale_y),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = (
            (
                hsv[:, :, 1]
                <= int(
                    self.settings.get(
                        "support_card_marker_max_saturation",
                        60,
                    )
                )
            )
            & (
                hsv[:, :, 2]
                >= int(
                    self.settings.get(
                        "support_card_marker_min_value",
                        170,
                    )
                )
            )
        ).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        roi_height, roi_width = white.shape
        best = 0.0
        for index in range(1, count):
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            area = int(stats[index, cv2.CC_STAT_AREA])
            width_ratio = width / max(1, roi_width)
            height_ratio = height / max(1, roi_height)
            aspect = width / max(1, height)
            area_ratio = area / max(1, roi_width * roi_height)
            if (
                width_ratio
                >= float(
                    self.settings.get(
                        "support_card_marker_min_width_ratio",
                        0.60,
                    )
                )
                and height_ratio
                <= float(
                    self.settings.get(
                        "support_card_marker_max_height_ratio",
                        0.75,
                    )
                )
                and aspect
                >= float(
                    self.settings.get(
                        "support_card_marker_min_aspect_ratio",
                        2.5,
                    )
                )
                and area_ratio
                >= float(
                    self.settings.get(
                        "support_card_marker_min_area_ratio",
                        0.14,
                    )
                )
            ):
                best = max(best, area_ratio)
        return best > 0.0, best

    def _card_affinity(
        self,
        screen: np.ndarray,
        point: tuple[int, int],
    ) -> tuple[str, float]:
        offset = self.settings.get(
            "card_affinity_roi_offset",
            [55, -210, 135, -150],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, point[0] + round(float(offset[0]) * scale_x))
        y1 = max(0, point[1] + round(float(offset[1]) * scale_y))
        x2 = min(
            screen.shape[1],
            point[0] + round(float(offset[2]) * scale_x),
        )
        y2 = min(
            screen.shape[0],
            point[1] + round(float(offset[3]) * scale_y),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return "neutral", 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue = (
            (hsv[:, :, 0] >= 85)
            & (hsv[:, :, 0] <= 125)
            & (hsv[:, :, 1] >= 80)
            & (hsv[:, :, 2] >= 80)
        )
        red = (
            ((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 170))
            & (hsv[:, :, 1] >= 90)
            & (hsv[:, :, 2] >= 90)
        )
        blue_ratio = float(np.count_nonzero(blue)) / float(blue.size)
        red_ratio = float(np.count_nonzero(red)) / float(red.size)
        minimum = float(
            self.settings.get("card_affinity_marker_min_ratio", 0.12)
        )
        if blue_ratio >= minimum and blue_ratio > red_ratio:
            return "disadvantage", blue_ratio
        if red_ratio >= minimum and red_ratio > blue_ratio:
            return "advantage", red_ratio
        return "neutral", max(blue_ratio, red_ratio)

    def _recognize_critical_digit(
        self,
        mask: np.ndarray,
    ) -> tuple[int, float]:
        glyph = _normalize_digit(mask)
        glyph_on = glyph > 0
        glyph_count = int(np.count_nonzero(glyph_on))
        best_digit = -1
        best_score = 0.0
        for digit, template in _critical_digit_templates():
            template_on = template > 0
            denominator = glyph_count + int(np.count_nonzero(template_on))
            if denominator == 0:
                continue
            score = (
                2.0
                * float(np.count_nonzero(glyph_on & template_on))
                / float(denominator)
            )
            if score > best_score:
                best_digit = digit
                best_score = score
        return best_digit, best_score

    def _critical_glyph_candidates(
        self,
        mask: np.ndarray,
    ) -> list[tuple[int, np.ndarray]]:
        """Extract the large outlined numeral glyphs from a crit label."""
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        max_component_area = int(
            self.settings.get("critical_max_component_area", 700)
        )
        parts = [
            tuple(int(value) for value in component)
            for component in stats[1:]
            if 4 <= int(component[4]) <= max_component_area
            and 4 <= int(component[1])
            and int(component[1]) + int(component[3])
            <= int(self.settings.get("critical_component_bottom", 64))
        ]
        anchors = [
            component
            for component in parts
            if int(component[4])
            >= int(self.settings.get("critical_anchor_min_area", 60))
            and int(component[3])
            >= int(self.settings.get("critical_anchor_min_height", 14))
            and int(component[2])
            <= int(self.settings.get("critical_anchor_max_width", 32))
            and 5 <= int(component[1]) <= 34
        ]
        candidates: list[tuple[int, np.ndarray]] = []
        for x, y, width, _, _ in sorted(anchors):
            related = []
            for component in parts:
                center_x = component[0] + component[2] / 2.0
                if (
                    x - 2 <= center_x <= x + width + 2
                    and component[1] >= y - 3
                ):
                    related.append(component)
            if not related:
                continue
            x1 = min(component[0] for component in related)
            y1 = min(component[1] for component in related)
            x2 = max(component[0] + component[2] for component in related)
            y2 = max(component[1] + component[3] for component in related)
            if (
                x2 - x1
                > int(self.settings.get("critical_glyph_max_width", 47))
                or y2 - y1
                < int(self.settings.get("critical_glyph_min_height", 23))
            ):
                continue
            if candidates and abs(x1 - candidates[-1][0]) <= 4:
                continue
            candidates.append((x1, mask[y1:y2, x1:x2]))
        return candidates

    def card_critical_percent(
        self,
        screen: np.ndarray,
        point: tuple[int, int],
    ) -> tuple[int | None, float]:
        """Read the 0%-100% critical probability displayed above a card."""
        offset = self.settings.get(
            "card_critical_roi_offset",
            [-90, -235, 100, -160],
        )
        scale_x = screen.shape[1] / self.base_width
        scale_y = screen.shape[0] / self.base_height
        x1 = max(0, point[0] + round(float(offset[0]) * scale_x))
        y1 = max(0, point[1] + round(float(offset[1]) * scale_y))
        x2 = min(
            screen.shape[1],
            point[0] + round(float(offset[2]) * scale_x),
        )
        y2 = min(
            screen.shape[0],
            point[1] + round(float(offset[3]) * scale_y),
        )
        roi = screen[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation_limits = self.settings.get(
            "critical_saturation_attempts",
            [60, 80, 100, 120, 150],
        )
        value_limits = self.settings.get(
            "critical_value_attempts",
            [100, 120, 140, 160, 180],
        )
        minimum = float(
            self.settings.get("critical_digit_min_confidence", 0.76)
        )
        best_value: int | None = None
        best_confidence = 0.0
        for saturation_limit in saturation_limits:
            for value_limit in value_limits:
                mask = np.where(
                    (hsv[:, :, 1] <= int(saturation_limit))
                    & (hsv[:, :, 2] >= int(value_limit)),
                    255,
                    0,
                ).astype(np.uint8)
                glyphs = self._critical_glyph_candidates(mask)
                if not glyphs:
                    continue
                recognized = [
                    (x, *self._recognize_critical_digit(glyph))
                    for x, glyph in glyphs
                ]
                first_x, first_digit, first_confidence = recognized[0]
                if first_digit < 0 or first_confidence < minimum:
                    continue
                value = first_digit * 10
                confidence = first_confidence

                # 100% is the only three-digit probability.  Confirm both
                # following zero glyphs; otherwise a leading 1 means 10%.
                if first_digit == 1:
                    following_zeros = [
                        (x, digit, digit_confidence)
                        for x, digit, digit_confidence in recognized[1:]
                        if x - first_x <= 85
                        and digit == 0
                        and digit_confidence >= minimum
                    ]
                    if len(following_zeros) >= 2:
                        value = 100
                        confidence = min(
                            confidence,
                            following_zeros[0][2],
                            following_zeros[1][2],
                        )
                if (
                    value == 100
                    and (
                        best_value != 100
                        or confidence > best_confidence
                    )
                ) or (
                    best_value != 100
                    and confidence > best_confidence
                ):
                    best_value = value
                    best_confidence = confidence
        return best_value, best_confidence

    def analyze_cards(self, screen: np.ndarray) -> list[CardInfo]:
        points = self.settings.get("card_points", {})
        cards: list[CardInfo] = []
        for index in range(1, 6):
            name = f"card{index}"
            if name not in points:
                raise BattleVisionError(f"战斗配置缺少 {name} 坐标")
            point = self._scaled_point(points[name], screen)
            color, confidence = self._card_color(screen, point)
            affinity, affinity_confidence = self._card_affinity(
                screen,
                point,
            )
            is_support, support_confidence = self._card_support_marker(
                screen,
                point,
            )
            critical_percent, critical_confidence = (
                self.card_critical_percent(screen, point)
            )
            is_disabled, disabled_confidence = self._card_disabled(
                screen,
                point,
            )
            cards.append(
                CardInfo(
                    name=name,
                    point=point,
                    color=color,
                    color_confidence=confidence,
                    affinity=affinity,
                    affinity_confidence=affinity_confidence,
                    owner_feature=self._owner_feature(screen, point),
                    is_support=is_support,
                    support_confidence=support_confidence,
                    critical_percent=critical_percent,
                    critical_confidence=critical_confidence,
                    is_disabled=is_disabled,
                    disabled_confidence=disabled_confidence,
                )
            )
        required = float(self.settings.get("card_color_min_confidence", 0.16))
        weak = [
            card.name
            for card in cards
            if not card.is_disabled and card.color_confidence < required
        ]
        if weak:
            raise BattleVisionError(
                f"指令卡颜色识别不稳定：{', '.join(weak)}"
            )
        return cards

    def owner_similarity(self, left: CardInfo, right: CardInfo) -> float:
        return float(np.dot(left.owner_feature, right.owner_feature))

    def _same_owner(self, cards: tuple[CardInfo, ...]) -> bool:
        threshold = float(self.settings.get("same_owner_threshold", 0.78))
        return all(
            self.owner_similarity(left, right) >= threshold
            for left, right in itertools.combinations(cards, 2)
        )

    def _primary_output_charge_bonus(
        self,
        cards: tuple[CardInfo, ...],
        *,
        primary_output_needs_charge: bool,
        primary_output_is_support: bool,
    ) -> tuple[float, int]:
        if not (
            primary_output_needs_charge
            and primary_output_is_support
        ):
            return 0.0, 0
        raw_weights = self.settings.get(
            "primary_output_charge_color_weights",
            {"arts": 12.0, "quick": 4.0, "buster": 0.0},
        )
        weights = {
            str(color): float(weight)
            for color, weight in raw_weights.items()
        }
        charge_cards = [
            card
            for card in cards
            if card.is_support and card.color in {"arts", "quick"}
        ]
        return (
            sum(
                weights.get(card.color, 0.0)
                for card in cards
                if card.is_support
            ),
            len(charge_cards),
        )

    def choose_face_cards(
        self,
        cards: list[CardInfo],
        *,
        charge_priority: bool,
        prefer_same_owner: bool = True,
        prefer_same_color: bool = True,
        prefer_arts_chain: bool = False,
        prefer_mighty_chain: bool = True,
        prefer_class_advantage: bool = True,
        prefer_primary_output: bool = False,
        primary_output_is_support: bool = False,
        prefer_damage_role_over_class: bool = False,
        prefer_high_critical: bool = True,
        primary_output_needs_charge: bool = False,
    ) -> CardPlan:
        color_weight = (
            {"arts": 2.8, "buster": 2.2, "quick": 1.3}
            if charge_priority
            else {"buster": 3.0, "arts": 2.1, "quick": 1.5}
        )
        has_disadvantage_cards = any(
            card.affinity == "disadvantage" for card in cards
        )
        damage_role_priority = (
            prefer_damage_role_over_class
            and prefer_primary_output
            and primary_output_is_support
        )
        output_card_weight = float(
            self.settings.get(
                (
                    "damage_role_card_weight"
                    if damage_role_priority
                    else "primary_output_card_weight"
                ),
                24.0 if damage_role_priority else 14.0,
            )
        )
        advantage_weight = float(
            self.settings.get(
                "damage_role_class_advantage_weight",
                4.0,
            )
            if damage_role_priority
            else 5.5
        )
        disadvantage_penalty = float(
            self.settings.get(
                "damage_role_class_disadvantage_penalty",
                9.0,
            )
            if damage_role_priority
            else 5.0
        )
        enabled_count = sum(not card.is_disabled for card in cards)
        minimum_disabled = max(0, 3 - enabled_count)
        ranked: list[tuple[float, tuple[CardInfo, ...], str]] = []
        for combo in itertools.combinations(cards, 3):
            disabled_count = sum(card.is_disabled for card in combo)
            if disabled_count != minimum_disabled:
                continue
            same_owner = self._same_owner(combo)
            same_color = len({card.color for card in combo}) == 1
            score = sum(color_weight.get(card.color, 0) for card in combo)
            if prefer_high_critical:
                score += sum(
                    card.critical_percent or 0 for card in combo
                ) * float(
                    self.settings.get(
                        "critical_probability_tiebreak_weight",
                        0.0001,
                    )
                )
            reasons: list[str] = []
            output_count = (
                sum(card.is_support for card in combo)
                if prefer_primary_output and primary_output_is_support
                else 0
            )
            if output_count:
                score += output_count * output_card_weight
                reasons.append(f"主力输出卡{output_count}张")
                if damage_role_priority:
                    reasons.append("输出角色优先于一般职阶克制")
            charge_bonus, output_charge_count = (
                self._primary_output_charge_bonus(
                    combo,
                    primary_output_needs_charge=(
                        primary_output_needs_charge
                    ),
                    primary_output_is_support=primary_output_is_support,
                )
            )
            score += charge_bonus
            if output_charge_count:
                reasons.append(
                    f"主力宝具未满，优先其充能卡{output_charge_count}张"
                )
            if prefer_class_advantage:
                advantage_count = sum(
                    card.affinity == "advantage" for card in combo
                )
                disadvantage_count = sum(
                    card.affinity == "disadvantage" for card in combo
                )
                score += advantage_count * advantage_weight
                score -= disadvantage_count * disadvantage_penalty
                if advantage_count:
                    reasons.append(f"职阶克制卡{advantage_count}张")
                elif has_disadvantage_cards and not disadvantage_count:
                    reasons.append("避开职阶抵抗卡")
            if same_owner and prefer_same_owner:
                score += 8.0
                reasons.append("同一从者三连")
            if same_color and prefer_same_color:
                score += 5.5
                reasons.append(f"{combo[0].color} 同色链")
            if (
                prefer_arts_chain
                and same_color
                and combo[0].color == "arts"
            ):
                score += float(
                    self.settings.get("prefer_arts_chain_bonus", 10.0)
                )
                reasons.append("三蓝 Arts Chain 优先")
            if (
                prefer_mighty_chain
                and {card.color for card in combo}
                == {"buster", "arts", "quick"}
            ):
                score += 4.0
                reasons.append("三色 Mighty Chain")
            if disabled_count:
                reasons.append(f"可行动卡不足，禁用卡补位{disabled_count}张")
            ranked.append((score, combo, "、".join(reasons) or "综合收益最高"))
        _, combo, reason = max(ranked, key=lambda item: item[0])
        first_order = (
            {"arts": 0, "quick": 1, "buster": 2}
            if primary_output_needs_charge
            else (
                {"arts": 0, "buster": 1, "quick": 2}
                if charge_priority
                else {"buster": 0, "arts": 1, "quick": 2}
            )
        )
        ordered = sorted(
            combo,
            key=lambda card: (
                card.is_disabled,
                (
                    0
                    if (
                        prefer_primary_output
                        and primary_output_is_support
                        and card.is_support
                    )
                    else 1
                ),
                first_order.get(card.color, 9),
            ),
        )
        return CardPlan(
            labels=[card.name for card in ordered],
            reason=reason,
            colors=[card.color for card in ordered],
        )

    def choose_with_np(
        self,
        cards: list[CardInfo],
        *,
        np_label: str,
        np_color: str | None,
        prefer_same_owner: bool = True,
        prefer_same_color: bool = True,
        prefer_arts_chain: bool = False,
        prefer_mighty_chain: bool = True,
        prefer_class_advantage: bool = True,
        prefer_primary_output: bool = False,
        primary_output_is_support: bool = False,
        prefer_damage_role_over_class: bool = False,
        prefer_high_critical: bool = True,
        primary_output_needs_charge: bool = False,
    ) -> CardPlan:
        weights = {"buster": 3.0, "arts": 2.2, "quick": 1.5}
        damage_role_priority = (
            prefer_damage_role_over_class
            and prefer_primary_output
            and primary_output_is_support
        )
        output_card_weight = float(
            self.settings.get(
                (
                    "damage_role_card_weight"
                    if damage_role_priority
                    else "primary_output_card_weight"
                ),
                24.0 if damage_role_priority else 14.0,
            )
        )
        advantage_weight = float(
            self.settings.get(
                "damage_role_class_advantage_weight",
                4.0,
            )
            if damage_role_priority
            else 5.5
        )
        disadvantage_penalty = float(
            self.settings.get(
                "damage_role_class_disadvantage_penalty",
                9.0,
            )
            if damage_role_priority
            else 5.0
        )
        enabled_count = sum(not card.is_disabled for card in cards)
        minimum_disabled = max(0, 2 - enabled_count)
        ranked: list[tuple[float, tuple[CardInfo, CardInfo], str]] = []
        for pair in itertools.combinations(cards, 2):
            disabled_count = sum(card.is_disabled for card in pair)
            if disabled_count != minimum_disabled:
                continue
            score = sum(weights.get(card.color, 0) for card in pair)
            if prefer_high_critical:
                score += sum(
                    card.critical_percent or 0 for card in pair
                ) * float(
                    self.settings.get(
                        "critical_probability_tiebreak_weight",
                        0.0001,
                    )
                )
            reasons = ["已就绪宝具优先"]
            output_count = (
                sum(card.is_support for card in pair)
                if prefer_primary_output and primary_output_is_support
                else 0
            )
            if output_count:
                score += output_count * output_card_weight
                reasons.append(f"主力输出卡{output_count}张")
                if damage_role_priority:
                    reasons.append("输出角色优先于一般职阶克制")
            charge_bonus, output_charge_count = (
                self._primary_output_charge_bonus(
                    pair,
                    primary_output_needs_charge=(
                        primary_output_needs_charge
                    ),
                    primary_output_is_support=primary_output_is_support,
                )
            )
            score += charge_bonus
            if output_charge_count:
                reasons.append(
                    f"主力宝具未满，优先其充能卡{output_charge_count}张"
                )
            if prefer_class_advantage:
                advantage_count = sum(
                    card.affinity == "advantage" for card in pair
                )
                disadvantage_count = sum(
                    card.affinity == "disadvantage" for card in pair
                )
                score += advantage_count * advantage_weight
                score -= disadvantage_count * disadvantage_penalty
                if advantage_count:
                    reasons.append(f"职阶克制卡{advantage_count}张")
                elif (
                    any(
                        card.affinity == "disadvantage"
                        for card in cards
                    )
                    and not disadvantage_count
                ):
                    reasons.append("避开职阶抵抗卡")
            if (
                prefer_same_color
                and np_color
                and all(card.color == np_color for card in pair)
            ):
                score += 7.0
                reasons.append(f"{np_color} 宝具同色链")
            if (
                prefer_arts_chain
                and np_color == "arts"
                and all(card.color == "arts" for card in pair)
            ):
                score += float(
                    self.settings.get("prefer_arts_chain_bonus", 10.0)
                )
                reasons.append("三蓝 Arts Chain 优先")
            if (
                prefer_mighty_chain
                and np_color
                and {np_color, pair[0].color, pair[1].color}
                == {"buster", "arts", "quick"}
            ):
                score += 4.0
                reasons.append("三色 Mighty Chain")
            if prefer_same_owner and self._same_owner(pair):
                score += 1.5
            if disabled_count:
                reasons.append(f"可行动卡不足，禁用卡补位{disabled_count}张")
            ranked.append((score, pair, "、".join(reasons)))
        _, pair, reason = max(ranked, key=lambda item: item[0])
        charge_order = {"arts": 0, "quick": 1, "buster": 2}
        pair = tuple(
            sorted(
                pair,
                key=lambda card: (
                    card.is_disabled,
                    (
                        0
                        if (
                            primary_output_needs_charge
                            and primary_output_is_support
                            and card.is_support
                        )
                        else 1
                    ),
                    charge_order.get(card.color, 9),
                ),
            )
        )
        return CardPlan(
            labels=[np_label, pair[0].name, pair[1].name],
            reason=reason,
            colors=[np_color or "np", pair[0].color, pair[1].color],
        )

    def choose_with_nps(
        self,
        cards: list[CardInfo],
        *,
        np_labels: list[str],
        prefer_same_owner: bool = True,
        prefer_same_color: bool = True,
        prefer_arts_chain: bool = False,
        prefer_mighty_chain: bool = True,
        prefer_class_advantage: bool = True,
        prefer_primary_output: bool = False,
        primary_output_is_support: bool = False,
        prefer_damage_role_over_class: bool = False,
        prefer_high_critical: bool = True,
        primary_output_needs_charge: bool = False,
    ) -> CardPlan:
        labels = list(np_labels[:3])
        colors = ["np"] * len(labels)
        if len(labels) < 3:
            face_plan = self.choose_face_cards(
                cards,
                charge_priority=primary_output_needs_charge,
                prefer_same_owner=prefer_same_owner,
                prefer_same_color=prefer_same_color,
                prefer_arts_chain=prefer_arts_chain,
                prefer_mighty_chain=prefer_mighty_chain,
                prefer_class_advantage=prefer_class_advantage,
                prefer_primary_output=prefer_primary_output,
                primary_output_is_support=primary_output_is_support,
                prefer_damage_role_over_class=(
                    prefer_damage_role_over_class
                ),
                prefer_high_critical=prefer_high_critical,
                primary_output_needs_charge=primary_output_needs_charge,
            )
            needed = 3 - len(labels)
            labels.extend(face_plan.labels[:needed])
            colors.extend(face_plan.colors[:needed])
        return CardPlan(
            labels=labels,
            reason=f"优先释放 {len(np_labels[:3])} 张已就绪宝具卡",
            colors=colors,
        )
