@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo   FGO 主线工具 一键修复脚本
echo   流程：检测 Python - 删除旧 .venv - 重建虚拟环境 - 装依赖 - 自检 - 启动
echo ============================================================
echo.

REM ==================== 位置检查 ====================
if not exist "requirements-lock.txt" (
    echo [错误] 未找到 requirements-lock.txt。
    echo 请把本脚本放到 FGO 工具根目录（与 启动控制面板.bat 同一文件夹）再运行。
    echo.
    pause
    exit /b 1
)
if not exist "wheels" (
    echo [错误] 未找到 wheels 目录，脚本位置不对或发布包不完整。
    echo.
    pause
    exit /b 1
)
if not exist "fgo_gui_launcher.pyw" (
    echo [错误] 缺少 fgo_gui_launcher.pyw，发布包不完整。
    echo.
    pause
    exit /b 1
)

REM ==================== 第 1 步：检测兼容 Python ====================
set "PY_CMD="
set "PY_VER="

python -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,12) and sys.maxsize>2**32 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
    goto :python_found
)

for %%V in (3.12 3.11 3.10) do (
    py -%%V -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,12) and sys.maxsize>2**32 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_CMD=py -%%V"
        for /f "tokens=*" %%v in ('py -%%V -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
        goto :python_found
    )
)

echo [错误] 未检测到兼容的 Python（需要 3.10 / 3.11 / 3.12 的 64 位版本）。
echo.
echo 请先安装 Python 3.12.10 后重新运行本脚本：
echo     https://www.python.org/downloads/release/python-31210/
echo 安装时请勾选 "Add python.exe to PATH"，装完直接双击本脚本即可。
echo.
pause
exit /b 1

:python_found
echo [1/5] 检测到 Python %PY_VER% （命令：%PY_CMD%）
echo.

REM ==================== 第 2 步：删除旧 .venv ====================
echo [2/5] 删除旧虚拟环境 .venv ...
if exist ".venv" rmdir /s /q ".venv"
if exist ".venv" (
    echo [错误] .venv 删除失败，可能有程序正在占用。
    echo 请关闭 FGO 控制面板窗口和所有 Python 相关进程后重试。
    echo.
    pause
    exit /b 1
)
echo       已删除
echo.

REM ==================== 第 3 步：重建虚拟环境 ====================
echo [3/5] 使用 %PY_VER% 重建虚拟环境 ...
%PY_CMD% -m venv .venv
if errorlevel 1 (
    echo [错误] 创建虚拟环境失败，请查看上方错误信息。
    echo.
    pause
    exit /b 1
)
echo       虚拟环境创建成功
echo.

REM ==================== 第 4 步：安装依赖 ====================
echo [4/5] 安装依赖（优先离线使用内置 wheels）...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --no-index --find-links wheels --only-binary=:all: --requirement requirements-lock.txt
if not errorlevel 1 goto :deps_ok
echo       离线安装失败，切换在线源：PyPI 官方 ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --retries 2 --timeout 20 --only-binary=:all: --index-url https://pypi.org/simple --requirement requirements-lock.txt
if not errorlevel 1 goto :deps_ok
echo       切换在线源：清华大学镜像 ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --retries 2 --timeout 20 --only-binary=:all: --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --requirement requirements-lock.txt
if not errorlevel 1 goto :deps_ok
echo       切换在线源：阿里云镜像 ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --retries 2 --timeout 20 --only-binary=:all: --index-url https://mirrors.aliyun.com/pypi/simple/ --requirement requirements-lock.txt
if not errorlevel 1 goto :deps_ok
echo [错误] 内置 wheels 和三个在线源均安装失败。
echo 请检查网络、防火墙或安全软件设置，然后重新运行本脚本。
echo.
pause
exit /b 1

:deps_ok
echo       依赖安装完成
echo.

REM ==================== 第 5 步：自检并启动 ====================
echo [5/5] 运行环境自检 ...
".venv\Scripts\python.exe" -c "import cv2, numpy, yaml, tkinter; print('Runtime check OK')"
if errorlevel 1 (
    echo [错误] 运行环境验证失败，请查看上方错误信息。
    echo.
    pause
    exit /b 1
)

REM 写入依赖指纹，与启动器保持一致，避免下次启动重复安装
powershell.exe -NoLogo -NoProfile -Command "$h=(Get-FileHash -LiteralPath 'requirements-lock.txt' -Algorithm SHA256).Hash; Set-Content -LiteralPath '.venv\.requirements.sha256' -Value $h -Encoding ASCII -NoNewline" >nul 2>&1

echo.
echo ============================================================
echo   修复完成！环境自检通过。
echo   正在打开控制面板 ...
echo ============================================================
start "" ".venv\Scripts\pythonw.exe" "fgo_gui_launcher.pyw"
pause
exit /b 0
