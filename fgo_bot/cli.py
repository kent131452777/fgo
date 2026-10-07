from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

from .adb import AdbError, MuMuDevice
from .calibration import (
    FAULT_ACTIONS,
    CalibrationCancelled,
    calibrate_battle_enemy_class,
    calibrate_enemy_class,
    calibrate_fault,
    calibrate_rule,
    calibrate_support_candidate,
)
from .class_affinity import ICON_CLASS_NAMES
from .config import (
    ConfigError,
    ensure_user_config,
    load_config,
    resolve_from_config,
    save_config,
)
from .runner import BotRunner, PauseRequested
from .selfcheck import (
    build_fault_matcher,
    classify_fault,
    load_selfcheck_settings,
)
from .strategy import SupportSelector
from .vision import (
    TemplateMatcher,
    VisionError,
    annotate_match,
    read_image,
    write_image,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "profiles" / "fgo_cn_1600x900.yaml"
USER_CONFIG = DEFAULT_CONFIG.with_name("user.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuMu 模拟器 FGO 保守型主线自动化",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML 配置文件路径；省略时使用 profiles/user.yaml",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("gui", help="打开图形控制面板")
    commands.add_parser("doctor", help="检查 MuMu、ADB、分辨率和模板")

    snapshot = commands.add_parser("snapshot", help="保存当前模拟器截图")
    snapshot.add_argument("--output", default="current-screen.png")

    rules = commands.add_parser("rules", help="列出规则和标定状态")
    rules.add_argument("--scores", action="store_true", help="同时对当前画面评分")

    calibrate = commands.add_parser("calibrate", help="框选并保存某个识别模板")
    calibrate.add_argument("--rule", required=True, help="规则名称")
    calibrate.add_argument("--input", help="从已有截图标定，而不是实时截图")

    enemy_class = commands.add_parser(
        "calibrate-enemy-class",
        help="标定关卡详情中的敌方职阶图标",
    )
    enemy_class.add_argument(
        "--class",
        dest="class_name",
        choices=ICON_CLASS_NAMES,
        required=True,
    )
    enemy_class.add_argument("--input", help="从已有截图标定")

    battle_enemy_class = commands.add_parser(
        "calibrate-battle-class",
        help="标定战斗界面中的敌方职阶图标",
    )
    battle_enemy_class.add_argument(
        "--class",
        dest="class_name",
        choices=ICON_CLASS_NAMES,
        required=True,
    )
    battle_enemy_class.add_argument("--input", help="从已有战斗截图标定")

    support = commands.add_parser(
        "calibrate-support",
        help="登记一个宝具5全体宝具助战候选",
    )
    support.add_argument("--id", dest="candidate_id", required=True)
    support.add_argument("--name", required=True)
    support.add_argument(
        "--class",
        dest="class_name",
        choices=ICON_CLASS_NAMES,
        required=True,
    )
    support.add_argument(
        "--np-color",
        choices=("buster", "arts", "quick"),
        required=True,
    )
    support.add_argument("--level", type=int, default=120)
    support.add_argument("--priority", type=int, default=0)
    support.add_argument("--party-slot", type=int, choices=(1, 2, 3), default=3)
    support.add_argument("--input", help="从已有助战列表截图标定")

    match = commands.add_parser("match", help="只识别一次并保存标注图")
    match.add_argument("--input", help="已有截图；省略则读取 MuMu")
    match.add_argument("--output", default="match-result.png")

    fill_roi = commands.add_parser(
        "fill-roi",
        help=(
            "扫描截图目录，为缺失 search_roi 的规则自动生成搜索区域"
            "并写回配置"
        ),
    )
    fill_roi.add_argument(
        "--input-dir",
        default="runs",
        help="截图目录（默认 runs，DRY-* 与 PAUSED 截图都会用）",
    )
    fill_roi.add_argument(
        "--online",
        action="store_true",
        help="额外用当前 MuMu 画面补一张截图参与定位",
    )

    selfcheck = commands.add_parser(
        "selfcheck",
        help="对当前画面运行自检：黑屏检测 + 故障模板分类报告",
    )
    selfcheck.add_argument("--input", help="已有截图；省略则读取 MuMu")

    fault = commands.add_parser(
        "calibrate-fault",
        help="框选并保存一个故障画面模板（写入 selfcheck.faults）",
    )
    fault.add_argument(
        "--name",
        required=True,
        help="故障名（英文、数字、下划线或连字符）",
    )
    fault.add_argument(
        "--action",
        choices=sorted(FAULT_ACTIONS),
        required=True,
        help="识别到该故障时执行的修正动作",
    )
    fault.add_argument(
        "--point",
        help="tap_point 动作的点击基准坐标，格式 x,y",
    )
    fault.add_argument("--threshold", type=float, help="匹配阈值（默认 0.88）")
    fault.add_argument("--input", help="从已有截图标定")

    run = commands.add_parser("run", help="启动状态机")
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="只识别，不点击")
    mode.add_argument("--live", action="store_true", help="允许实际点击")
    run.add_argument(
        "--mode",
        choices=("story", "farm"),
        default=None,
        help="覆盖配置文件中的运行模式（story=主线推进 / farm=日常刷本）",
    )
    return parser


def _build_device(config: dict[str, Any]) -> MuMuDevice:
    device = config["device"]
    return MuMuDevice(
        device.get("manager_path"),
        int(device.get("vm_index", 0)),
    )


def _show_rules(
    config: dict[str, Any],
    config_path: Path,
    matcher: TemplateMatcher,
) -> None:
    print("规则状态（配置顺序即识别优先级）：")
    available = set(matcher.available_rules)
    for rule in config["rules"]:
        path = resolve_from_config(config_path, rule["template"])
        if not rule.get("enabled", True):
            status = "已禁用"
        elif rule["name"] in available:
            status = "可用"
        else:
            status = "缺少模板"
        print(
            f"  {rule['name']:<22} {status:<8} "
            f"{rule['action']:<10} {path}"
        )


def _doctor(
    device: MuMuDevice,
    config: dict[str, Any],
    config_path: Path,
    matcher: TemplateMatcher,
) -> int:
    info = device.connect()
    print(f"MuMu：{info.name}（实例 {info.vm_index}）")
    print(f"Android：{info.android_version}；状态：{info.player_state}")
    print(f"ADB：{info.serial}")
    screen = device.capture()
    height, width = screen.shape[:2]
    expected = (
        int(config["screen"]["base_width"]),
        int(config["screen"]["base_height"]),
    )
    print(f"截图：{width}x{height}；配置基准：{expected[0]}x{expected[1]}")
    package = device.foreground_package()
    wanted = config["device"].get("package")
    print(f"前台包名：{package or '未知'}")
    if wanted and package != wanted:
        print(f"警告：预期包名为 {wanted}")
    scores = matcher.scores(screen)[:5]
    if scores:
        print("当前画面匹配：")
        for name, score in scores:
            print(f"  {name:<22} {score:.3f}")
    selector = SupportSelector(config, config_path)
    print(
        f"敌方职阶模板：{len(selector.enemy_templates)}；"
        f"合格助战候选：{selector.qualified_candidate_count}"
    )
    _show_rules(config, config_path, matcher)
    settings = load_selfcheck_settings(config_path)
    if settings is None:
        print("自检：配置读取失败")
    else:
        _, selfcheck = settings
        print(
            f"自检：{'已启用' if selfcheck.get('enabled') else '已关闭'}；"
            f"故障模板 {len(selfcheck.get('faults') or [])} 个；"
            f"失败兜底 "
            f"{'自动重启游戏' if selfcheck.get('restart_app') else '暂停等待人工'}"
        )
    if (width, height) != expected:
        print("提示：分辨率不一致时会自动缩放模板，但重新标定更可靠。")
    return 0


def _selfcheck_report(
    device: MuMuDevice,
    config: dict[str, Any],
    config_path: Path,
    input_path: Path | None,
) -> int:
    """对当前（或指定）画面跑一次自检分类并打印报告。"""
    loaded = load_selfcheck_settings(config_path)
    if loaded is None:
        print("自检配置读取失败", file=sys.stderr)
        return 1
    _, settings = loaded
    print("自检设置：")
    for key in (
        "enabled",
        "max_attempts",
        "max_consecutive_recoveries",
        "restart_app",
        "restart_emulator",
        "back_press_first",
        "black_screen_max_mean",
    ):
        print(f"  {key}: {settings.get(key)}")
    faults = settings.get("faults") or []
    print(f"  故障模板数：{len(faults)}")
    screen = read_image(input_path) if input_path else device.capture()
    fault_matcher = build_fault_matcher(config, config_path, settings)
    name, rule, match = classify_fault(screen, settings, fault_matcher)
    if name is not None:
        print(
            f"当前画面识别为故障：{name}；"
            f"将执行动作：{(rule or {}).get('action', '')}"
        )
        if match is not None:
            print(f"  置信度 {match.score:.3f}；中心 {match.center}")
    else:
        print("当前画面未识别为故障")
    if faults:
        print("各故障规则得分（调阈值参考）：")
        for rule_name, score in fault_matcher.scores(screen):
            print(f"  {rule_name:<24} {score:.3f}")
    return 0


def _fill_roi(
    config: dict[str, Any],
    config_path: Path,
    matcher: TemplateMatcher,
    device: MuMuDevice,
    input_dir: Path,
    online: bool,
) -> int:
    """用历史截图定位缺 search_roi 规则的模板中心，自动生成搜索区域。

    位置来自截图中的实际命中中心（中位数 + 集中度校验），
    大模板（>ROI 尺寸）跳过避免规则失效。
    """
    screen_cfg = config["screen"]
    base_w = int(screen_cfg["base_width"])
    base_h = int(screen_cfg["base_height"])

    screens: list[tuple[str, Any]] = []
    if input_dir.is_dir():
        for png in sorted(input_dir.glob("*.png")):
            try:
                screens.append((png.name, read_image(png)))
            except VisionError:
                continue
    if online:
        try:
            screens.append(("live", device.capture()))
        except AdbError as exc:
            print(f"在线截图失败（MuMu 未连接？）：{exc}", file=sys.stderr)

    missing = [rule for rule in config["rules"] if not rule.get("search_roi")]
    if not missing:
        print("所有规则都已配置 search_roi，无需处理")
        return 0
    print(f"缺 search_roi 的规则：{len(missing)} 条；分析截图：{len(screens)} 张")

    filled: list[tuple[str, list[int], int]] = []
    skipped: list[tuple[str, str]] = []
    for rule in missing:
        name = str(rule["name"])
        if name not in matcher.available_rules:
            skipped.append((name, "模板缺失"))
            continue
        try:
            template = read_image(resolve_from_config(config_path, rule["template"]))
        except VisionError:
            skipped.append((name, "模板读取失败"))
            continue
        centers: list[tuple[int, int]] = []
        for _label, screen in screens:
            result = matcher.match_rule(screen, rule)
            if result is not None:
                # match_rule 返回的是截图分辨率坐标，而 search_roi 写回的是
                # 基准分辨率坐标（消费时按 screen/base 比例映射），必须换算。
                height, width = screen.shape[:2]
                cx, cy = result.center
                centers.append(
                    (
                        round(cx * base_w / width),
                        round(cy * base_h / height),
                    )
                )
        if not centers:
            skipped.append((name, "截图未命中"))
            continue
        xs = [point[0] for point in centers]
        ys = [point[1] for point in centers]
        if len(centers) >= 2:
            spread_x = statistics.pstdev(xs)
            spread_y = statistics.pstdev(ys)
            if spread_x > 150 or spread_y > 150:
                skipped.append(
                    (
                        name,
                        f"命中位置分散 (σx={spread_x:.0f}, σy={spread_y:.0f})",
                    )
                )
                continue
        center_x = round(statistics.median(xs))
        center_y = round(statistics.median(ys))
        margin = 300
        roi = [
            max(0, center_x - margin),
            max(0, center_y - margin),
            min(base_w, center_x + margin),
            min(base_h, center_y + margin),
        ]
        roi_w = roi[2] - roi[0]
        roi_h = roi[3] - roi[1]
        template_h, template_w = template.shape[:2]
        if roi_w < 100 or roi_h < 100:
            skipped.append((name, "ROI 过小"))
            continue
        if template_w > roi_w or template_h > roi_h:
            skipped.append(
                (
                    name,
                    f"模板 {template_w}x{template_h} 大于 ROI {roi_w}x{roi_h}，"
                    "需手动标定",
                )
            )
            continue
        rule["search_roi"] = roi
        filled.append((name, roi, len(centers)))

    save_config(config, config_path)
    for name, roi, count in filled:
        print(f"  写入 {name:<24} search_roi: {roi}（{count} 张截图命中）")
    for name, reason in skipped:
        print(f"  跳过 {name:<24} {reason}")
    print(f"完成：写入 {len(filled)} 条，跳过 {len(skipped)} 条")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selected_config = (
            Path(args.config).expanduser().resolve()
            if args.config
            else ensure_user_config(DEFAULT_CONFIG, USER_CONFIG)
        )
        config, config_path = load_config(selected_config)
        device = _build_device(config)
        matcher = TemplateMatcher(config, config_path)

        if args.command == "gui":
            from .gui import launch_gui

            launch_gui(config, config_path, device)
            return 0

        if args.command == "doctor":
            return _doctor(device, config, config_path, matcher)

        if args.command == "snapshot":
            output = Path(args.output).expanduser().resolve()
            write_image(output, device.capture())
            print(f"已保存：{output}")
            return 0

        if args.command == "rules":
            _show_rules(config, config_path, matcher)
            if args.scores:
                for name, score in matcher.scores(device.capture()):
                    print(f"  score {name:<22} {score:.3f}")
            return 0

        if args.command == "calibrate":
            input_path = Path(args.input).resolve() if args.input else None
            target = calibrate_rule(
                device,
                config,
                config_path,
                args.rule,
                input_path=input_path,
            )
            print(f"标定完成：{target}")
            return 0

        if args.command == "calibrate-enemy-class":
            input_path = Path(args.input).resolve() if args.input else None
            target = calibrate_enemy_class(
                device,
                config,
                config_path,
                args.class_name,
                input_path=input_path,
            )
            print(f"敌方职阶标定完成：{target}")
            return 0

        if args.command == "calibrate-battle-class":
            input_path = Path(args.input).resolve() if args.input else None
            target = calibrate_battle_enemy_class(
                device,
                config,
                config_path,
                args.class_name,
                input_path=input_path,
            )
            print(f"战斗敌方职阶标定完成：{target}")
            return 0

        if args.command == "calibrate-support":
            input_path = Path(args.input).resolve() if args.input else None
            target = calibrate_support_candidate(
                device,
                config,
                config_path,
                candidate_id=args.candidate_id,
                name=args.name,
                class_name=args.class_name,
                np_color=args.np_color,
                servant_level=args.level,
                priority=args.priority,
                party_slot=args.party_slot,
                input_path=input_path,
            )
            print(f"助战候选登记完成：{target}")
            return 0

        if args.command == "match":
            screen = read_image(Path(args.input).resolve()) if args.input else device.capture()
            result = matcher.find_first(screen)
            output = Path(args.output).expanduser().resolve()
            if result is None:
                write_image(output, screen)
                print(f"未命中任何规则；原图保存为：{output}")
                return 2
            write_image(output, annotate_match(screen, result))
            print(
                f"命中 {result.rule['name']}，置信度 {result.score:.3f}；"
                f"标注图：{output}"
            )
            return 0

        if args.command == "selfcheck":
            input_path = Path(args.input).resolve() if args.input else None
            return _selfcheck_report(device, config, config_path, input_path)

        if args.command == "fill-roi":
            input_dir = Path(args.input_dir).expanduser().resolve()
            return _fill_roi(
                config,
                config_path,
                matcher,
                device,
                input_dir,
                bool(args.online),
            )

        if args.command == "calibrate-fault":
            input_path = Path(args.input).resolve() if args.input else None
            point = None
            if args.point:
                parts = [part.strip() for part in args.point.split(",")]
                if len(parts) != 2:
                    raise ValueError("--point 格式应为 x,y")
                point = [int(parts[0]), int(parts[1])]
            target = calibrate_fault(
                device,
                config,
                config_path,
                args.name,
                args.action,
                point=point,
                threshold=args.threshold,
                input_path=input_path,
            )
            print(f"故障标定完成：{target}")
            return 0

        if args.command == "run":
            if args.mode is not None:
                # 仅内存覆盖本次运行的模式，不写回配置文件。
                config.setdefault("behavior", {})["mode"] = args.mode
            runner = BotRunner(
                device,
                matcher,
                config,
                config_path,
                dry_run=bool(args.dry_run),
            )
            try:
                runner.run()
            except KeyboardInterrupt:
                runner.stop()
                print("\n收到 Ctrl+C，已停止。")
            return 0

        raise ConfigError(f"未知命令：{args.command}")
    except CalibrationCancelled as exc:
        print(f"标定已取消：{exc}", file=sys.stderr)
        return 2
    except PauseRequested as exc:
        print(f"已安全暂停：{exc.reason}", file=sys.stderr)
        if exc.snapshot:
            print(f"现场截图：{exc.snapshot}", file=sys.stderr)
        return 2
    except (AdbError, ConfigError, VisionError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
