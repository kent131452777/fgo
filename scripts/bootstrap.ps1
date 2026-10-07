[CmdletBinding()]
param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDirectory = Join-Path $ProjectRoot "logs"
$LogPath = Join-Path $LogDirectory "setup.log"
$VenvDirectory = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"
$VenvPythonw = Join-Path $VenvDirectory "Scripts\pythonw.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements-lock.txt"
$StampPath = Join-Path $VenvDirectory ".requirements.sha256"
$LauncherPath = Join-Path $ProjectRoot "fgo_gui_launcher.pyw"
$WheelDirectory = Join-Path $ProjectRoot "wheels"
$WheelManifestPath = Join-Path $WheelDirectory "SHA256SUMS.txt"

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

function Write-SetupLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

function Invoke-LoggedNative {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [switch]$AllowFailure
    )
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $FilePath @Arguments 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    }
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($nativeExitCode -ne 0) {
        if ($AllowFailure) {
            return $nativeExitCode
        }
        throw "命令执行失败（代码 $nativeExitCode）：$FilePath $($Arguments -join ' ')"
    }
    if ($AllowFailure) {
        return 0
    }
}

function Test-Wheelhouse {
    if (-not (Test-Path -LiteralPath $WheelManifestPath -PathType Leaf)) {
        return $false
    }
    $verified = 0
    foreach ($line in Get-Content -LiteralPath $WheelManifestPath) {
        if ($line -notmatch '^([0-9a-fA-F]{64})\s{2,}(.+)$') {
            continue
        }
        $expectedHash = $matches[1].ToLowerInvariant()
        $wheelPath = Join-Path $WheelDirectory $matches[2]
        if (-not (Test-Path -LiteralPath $wheelPath -PathType Leaf)) {
            return $false
        }
        $actualHash = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            throw "内置依赖校验失败：$wheelPath"
        }
        $verified += 1
    }
    return $verified -ge 7
}

function Install-RuntimeDependencies {
    if (Test-Wheelhouse) {
        Write-SetupLog "正在安装发布包内置依赖，无需连接 PyPI"
        $offlineExitCode = Invoke-LoggedNative -FilePath $VenvPython -AllowFailure -Arguments @(
            "-m", "pip", "install", "--disable-pip-version-check",
            "--no-index", "--find-links", $WheelDirectory,
            "--only-binary=:all:", "--requirement", $RequirementsPath
        )
        if ($offlineExitCode -eq 0) {
            return
        }
        Write-SetupLog "内置依赖与当前Python不匹配，开始尝试网络源"
    }
    else {
        Write-SetupLog "发布包未包含完整离线依赖，开始尝试网络源"
    }

    $indexes = @(
        @("Python官方PyPI", "https://pypi.org/simple"),
        @("清华大学PyPI镜像", "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"),
        @("阿里云PyPI镜像", "https://mirrors.aliyun.com/pypi/simple/")
    )
    foreach ($index in $indexes) {
        Write-SetupLog "尝试依赖源：$($index[0])"
        $onlineExitCode = Invoke-LoggedNative -FilePath $VenvPython -AllowFailure -Arguments @(
            "-m", "pip", "install", "--disable-pip-version-check",
            "--retries", "2", "--timeout", "20", "--only-binary=:all:",
            "--index-url", $index[1], "--requirement", $RequirementsPath
        )
        if ($onlineExitCode -eq 0) {
            return
        }
        Write-SetupLog "$($index[0])安装失败，自动切换下一个依赖源"
    }
    throw "内置依赖及三个网络源均安装失败。请查看日志并检查网络、防火墙或安全软件。"
}

function Get-CompatiblePython {
    $python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($python) {
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & $python.Source -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) and sys.maxsize > 2**32 else 1)" 2>$null
        $pythonExitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorAction
        if ($pythonExitCode -eq 0) {
            return [pscustomobject]@{
                Executable = $python.Source
                Prefix = @()
            }
        }
    }

    $launcher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($version in @("3.12", "3.11", "3.10")) {
            $previousErrorAction = $ErrorActionPreference
            $ErrorActionPreference = "SilentlyContinue"
            & $launcher.Source "-$version" -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) and sys.maxsize > 2**32 else 1)" 2>$null
            $launcherExitCode = $LASTEXITCODE
            $ErrorActionPreference = $previousErrorAction
            if ($launcherExitCode -eq 0) {
                return [pscustomobject]@{
                    Executable = $launcher.Source
                    Prefix = @("-$version")
                }
            }
        }
    }

    return $null
}

try {
    Write-SetupLog "检查 FGO 主线工具运行环境"
    foreach ($requiredFile in @($RequirementsPath, $LauncherPath)) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            throw "发布包缺少文件：$requiredFile"
        }
    }

    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        $hostPython = Get-CompatiblePython
        if (-not $hostPython) {
            throw "未找到兼容的 64 位 Python。请安装 Python 3.10、3.11 或 3.12 后重新双击启动控制面板.bat。推荐下载：https://www.python.org/downloads/release/python-31210/"
        }
        $hostExecutable = $hostPython.Executable
        $hostArguments = @($hostPython.Prefix)
        $hostArguments += @("-m", "venv", $VenvDirectory)
        Write-SetupLog "首次运行：正在创建独立环境 .venv"
        Invoke-LoggedNative -FilePath $hostExecutable -Arguments $hostArguments
    }

    $requirementsHash = (Get-FileHash -LiteralPath $RequirementsPath -Algorithm SHA256).Hash
    $installedHash = ""
    if (Test-Path -LiteralPath $StampPath -PathType Leaf) {
        $installedHash = (Get-Content -LiteralPath $StampPath -Raw).Trim()
    }

    if ($requirementsHash -ne $installedHash) {
        Install-RuntimeDependencies
        Set-Content -LiteralPath $StampPath -Value $requirementsHash -Encoding ASCII
        Write-SetupLog "依赖安装完成"
    }
    else {
        Write-SetupLog "运行依赖已就绪，无需重复下载"
    }

    Invoke-LoggedNative -FilePath $VenvPython -Arguments @(
        "-c", "import cv2, numpy, yaml, tkinter; print('Runtime check OK')"
    )

    if ($CheckOnly) {
        Write-SetupLog "环境检查通过"
        exit 0
    }

    if (-not (Test-Path -LiteralPath $VenvPythonw -PathType Leaf)) {
        throw "虚拟环境缺少 pythonw.exe：$VenvPythonw"
    }
    Write-SetupLog "正在打开控制面板"
    Start-Process -FilePath $VenvPythonw -ArgumentList @("`"$LauncherPath`"") -WorkingDirectory $ProjectRoot
    exit 0
}
catch {
    Write-SetupLog "失败：$($_.Exception.Message)"
    Write-Host ""
    Write-Host "环境准备失败。详细日志：$LogPath" -ForegroundColor Red
    exit 1
}
