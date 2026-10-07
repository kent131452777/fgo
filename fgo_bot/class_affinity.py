from __future__ import annotations

from collections.abc import Iterable


# 官方常规职阶相性。Beast 的相性会随具体灵基变化，因此 generic "beast"
# 不做推断；已公开的特殊灵基使用独立名称表示。
ATTACK_ADVANTAGE: dict[str, frozenset[str]] = {
    "saber": frozenset({"lancer"}),
    "archer": frozenset({"saber"}),
    "lancer": frozenset({"archer"}),
    "rider": frozenset({"caster"}),
    "caster": frozenset({"assassin"}),
    "assassin": frozenset({"rider"}),
    "berserker": frozenset(
        {
            "saber",
            "archer",
            "lancer",
            "rider",
            "caster",
            "assassin",
            "berserker",
            "ruler",
            "avenger",
            "moon_cancer",
            "alter_ego",
            "pretender",
        }
    ),
    "ruler": frozenset({"moon_cancer"}),
    "avenger": frozenset({"ruler"}),
    "moon_cancer": frozenset({"avenger"}),
    "alter_ego": frozenset(
        {"rider", "caster", "assassin", "foreigner"}
    ),
    "foreigner": frozenset({"berserker", "foreigner", "pretender"}),
    "pretender": frozenset(
        {"saber", "archer", "lancer", "alter_ego"}
    ),
    "shielder": frozenset(),
    "beast": frozenset(),
    "beast_draco": frozenset(
        {
            "saber",
            "archer",
            "lancer",
            "rider",
            "caster",
            "assassin",
            "berserker",
        }
    ),
    "beast_space_ereshkigal": frozenset(
        {
            "ruler",
            "alter_ego",
            "moon_cancer",
            "foreigner",
            "pretender",
        }
    ),
    "unbeast": frozenset({"moon_cancer", "berserker", "foreigner"}),
}


# 用于选助战的优先顺序：先选完整克制关系，再由调用方决定是否使用狂阶兜底。
COUNTER_PRIORITY: dict[str, tuple[str, ...]] = {
    "saber": ("archer",),
    "archer": ("lancer",),
    "lancer": ("saber",),
    "rider": ("assassin",),
    "caster": ("rider",),
    "assassin": ("caster",),
    "berserker": ("foreigner", "berserker"),
    "ruler": ("avenger",),
    "avenger": ("moon_cancer",),
    "moon_cancer": ("ruler",),
    "alter_ego": ("pretender",),
    "foreigner": ("alter_ego",),
    "pretender": ("foreigner",),
    "shielder": (),
    "beast": (),
    "beast_draco": (
        "ruler",
        "avenger",
        "alter_ego",
        "moon_cancer",
        "foreigner",
        "pretender",
    ),
    "beast_space_ereshkigal": ("avenger",),
    "unbeast": ("avenger", "berserker", "foreigner"),
}


CLASS_NAMES = tuple(ATTACK_ADVANTAGE)
ICON_CLASS_NAMES = (
    "saber",
    "archer",
    "lancer",
    "rider",
    "caster",
    "assassin",
    "berserker",
    "ruler",
    "avenger",
    "moon_cancer",
    "alter_ego",
    "foreigner",
    "pretender",
    "shielder",
    "beast",
    "unbeast",
)


def has_attack_advantage(
    attacker_class: str | None,
    defender_class: str | None,
) -> bool:
    if not attacker_class or not defender_class:
        return False
    return defender_class in ATTACK_ADVANTAGE.get(
        str(attacker_class),
        frozenset(),
    )


def counter_classes(
    enemy_class: str | None,
    overrides: dict[str, str | Iterable[str]] | None = None,
) -> list[str]:
    if not enemy_class:
        return []
    if overrides and enemy_class in overrides:
        raw = overrides[enemy_class]
        if isinstance(raw, str):
            return [raw]
        return [str(value) for value in raw]
    return list(COUNTER_PRIORITY.get(enemy_class, ()))
