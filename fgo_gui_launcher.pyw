from __future__ import annotations

import ctypes
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from fgo_bot.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "logs" / "control-panel-startup.log"


def _show_startup_error() -> None:
    message = f"控制面板启动失败。\n错误日志：{LOG_PATH}"
    ctypes.windll.user32.MessageBoxW(
        0,
        message,
        "FGO 主线自动推进",
        0x10,
    )


def launch() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(
            f"\n[{datetime.now().isoformat(timespec='seconds')}] 启动控制面板\n"
        )
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                exit_code = int(main(["gui"]))
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 0
        except Exception:
            traceback.print_exc(file=log_file)
            _show_startup_error()
            return 1
        if exit_code != 0:
            log_file.write(f"控制面板返回错误代码：{exit_code}\n")
            _show_startup_error()
        return exit_code


if __name__ == "__main__":
    raise SystemExit(launch())
