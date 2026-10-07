from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def ensure_user_config(default_path: Path, user_path: Path) -> Path:
    """Create an editable local config without changing the shipped profile."""
    if user_path.is_file():
        return user_path.resolve()
    if not default_path.is_file():
        raise ConfigError(f"默认配置文件不存在：{default_path}")
    user_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(default_path, user_path)
    return user_path.resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"配置文件不存在：{config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    required = ("device", "screen", "behavior", "rules")
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"配置缺少字段：{', '.join(missing)}")
    if not isinstance(data["rules"], list) or not data["rules"]:
        raise ConfigError("rules 必须是非空列表")

    names: set[str] = set()
    for index, rule in enumerate(data["rules"]):
        if not isinstance(rule, dict):
            raise ConfigError(f"rules[{index}] 不是对象")
        name = str(rule.get("name", "")).strip()
        if not name:
            raise ConfigError(f"rules[{index}] 缺少 name")
        if name in names:
            raise ConfigError(f"规则名称重复：{name}")
        names.add(name)
        if not rule.get("template"):
            raise ConfigError(f"规则 {name} 缺少 template")
        if not rule.get("action"):
            raise ConfigError(f"规则 {name} 缺少 action")

    return data, config_path


def save_config(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(
            data,
            stream,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )


def resolve_from_config(config_path: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def get_rule(config: dict[str, Any], name: str) -> dict[str, Any]:
    for rule in config["rules"]:
        if rule["name"] == name:
            return rule
    available = ", ".join(rule["name"] for rule in config["rules"])
    raise ConfigError(f"没有规则 {name}；可用规则：{available}")
