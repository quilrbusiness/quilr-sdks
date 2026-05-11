#Requires -Version 5.1
# =============================================================================
# local-runner.ps1  -  DevSecOps Local Stage Runner (Windows)
#
# Mirrors .github/workflows/ci.yml exactly: each stage runs inside the same
# Docker image used by CI, with the project and ci-scripts bind-mounted so
# the environment is bit-for-bit identical to what GitHub Actions sees.
#
# Usage:  Get-Help .\local-runner.ps1 -Full
#         .\local-runner.ps1 --help
# =============================================================================

[CmdletBinding()]
param(
    [Alias('s')][string[]]$Stage       = @(),
    [Alias('d')][string]  $ProjectDir  = (Get-Location).Path,
    [Alias('c')][string]  $CiScriptsDir = "",
    # Sonar/SAST removed - no token required
    [Alias('o')][string]  $Org         = $(if ($env:GITHUB_ORG) { $env:GITHUB_ORG } else { "quilrbusiness" }),
    [Alias('l')][string]  $LogLevel    = "ERROR",
    [string]              $Language    = "",
    [string]              $OutputDir   = "",
    [switch]              $Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Try to configure the PowerShell host to emit UTF-8 so Unicode glyphs display correctly
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
    # Attempt to set the console code page to UTF-8; ignore failures
    cmd.exe /c chcp 65001 > $null 2>&1
} catch { }

# Attempt to repair common mojibake sequences by re-decoding the string bytes
function TryFix-Mojibake {
    param([string]$s)
    if (-not $s) { return $s }
    try {
        $enc1252 = [System.Text.Encoding]::GetEncoding(1252)
        $bytes = $enc1252.GetBytes($s)
        $candidate = [System.Text.Encoding]::UTF8.GetString($bytes)

        # If candidate contains any non-ASCII character, prefer it
        if ($candidate.ToCharArray() | Where-Object { [int]$_ -gt 127 }) { return $candidate }

        # Prefer the candidate if it contains fewer replacement chars
        $origBad = ($s.ToCharArray() | Where-Object { [int]$_ -eq 65533 }).Count
        $candBad = ($candidate.ToCharArray() | Where-Object { [int]$_ -eq 65533 }).Count
        if ($candBad -lt $origBad) { return $candidate }
    } catch { }
    return $s
}

function Normalize-ConsoleString {
    param([string]$s)
    if (-not $s) { return $s }
    $fixed = TryFix-Mojibake $s
    return $fixed
}

# ── Colors ───────────────────────────────────────────────────────────────────
$script:ColBlue   = "Cyan"
$script:ColGreen  = "Green"
$script:ColYellow = "Yellow"
$script:ColRed    = "Red"
$script:ColCyan   = "Cyan"

# ── Logger ───────────────────────────────────────────────────────────────────
function _ts { Get-Date -Format "yyyy-MM-dd HH:mm:ss" }

function _log {
    param([string]$Level, [string]$Color, [string]$Msg, [switch]$Stderr)
    $padded = "[$Level]".PadRight(9)
    $line   = "$padded $(_ts) - $Msg"
    $out = Normalize-ConsoleString $line
    if ($Stderr) { [Console]::Error.WriteLine($out) }
    else         { Write-Host $out -ForegroundColor $Color }
}

function log_debug { if ($script:LogLevel -eq "DEBUG") { _log "DEBUG"   $script:ColBlue   $args[0] -Stderr } }
function log_info  { _log "INFO"    $script:ColBlue   $args[0] }
function log_warn  { _log "WARN"    $script:ColYellow $args[0] }
function log_error { _log "ERROR"   $script:ColRed    $args[0] -Stderr }
function log_ok    { _log "SUCCESS" $script:ColGreen  $args[0] }
function log_fail  { log_error $args[0]; exit 1 }

$script:LogLevel = $LogLevel.ToUpper()

# ── Constants ────────────────────────────────────────────────────────────────
$script:SCRIPT_DIR        = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:WORKSPACE_MOUNT   = "/opt/workspace"
$script:REGISTRY          = "nexus.quilr.ai/ci-oss-images"
$script:CI_SCRIPTS_REPO   = "ci-scripts"
$script:CI_SCRIPTS_BRANCH = "ci-security-tools-integration"
$script:OUTPUT_SUBDIR     = "devsecops-precheck-reports"
$script:IMG_SECRET_SCAN   = "ci-security-tools:python-3-12-slim-secret-scan"
$script:IMG_SCA           = "ci-security-tools:ubuntu-24-04-security-tools-sca"
$script:IMG_IAC           = "ci-security-tools:ubuntu-24-04-security-tools-iac"
$script:ALL_STAGES        = @("secret-scan", "sca", "iac", "linter")
$script:STRAY_REPORTS     = @(
    "gitleaks-report.html",    "gitleaks-report.json",
    "trivy-report.html",       "trivy-report.json",
    "shellcheck-report.html",  "shellcheck-report.xml",
    "shellcheck-report.json",  "checkstyle-pom.xml",
    "eslint-report.html",      "eslint-report.json",
    "flake8-report-html",      "flake8-report.json",
    "checkstyle-report-html",  "checkstyle-report.xml",
    "clippy-report.html",      "clippy-report.xml",
    "clippy-report.json",      "counts.json",
    "ci-scripts",              "artifacts",
    "eslint.config.js",        "checkstyle.xml",
    "target",                  ".gitleaks.toml"
)

log_debug "SCRIPT_DIR        : $($script:SCRIPT_DIR)"
log_debug "WORKSPACE_MOUNT   : $($script:WORKSPACE_MOUNT)"
log_debug "REGISTRY          : $($script:REGISTRY)"
log_debug "CI_SCRIPTS_REPO   : $($script:CI_SCRIPTS_REPO)"
log_debug "CI_SCRIPTS_BRANCH : $($script:CI_SCRIPTS_BRANCH)"
log_debug "OUTPUT_SUBDIR     : $($script:OUTPUT_SUBDIR)"
log_debug "IMG_SECRET_SCAN   : $($script:IMG_SECRET_SCAN)"
log_debug "IMG_SCA           : $($script:IMG_SCA)"
log_debug "IMG_IAC           : $($script:IMG_IAC)"

# ── Mutable globals ──────────────────────────────────────────────────────────
$script:CiScriptsDir   = $CiScriptsDir
$script:ProjectDir     = $ProjectDir
$script:ProjectName    = ""
$script:OutputDir      = $OutputDir
$script:Org            = $Org
$script:LinterLanguage = $Language
$script:Stages         = [System.Collections.Generic.List[string]]::new()
$script:StageResults   = @{}
$script:CiScriptsTmp   = ""

foreach ($s in $Stage) { $script:Stages.Add($s) }

log_debug "ProjectDir     : $($script:ProjectDir)"
log_debug "OutputDir      : $($script:OutputDir)"
log_debug "Org            : $($script:Org)"
log_debug "LinterLanguage : $($script:LinterLanguage)"

# If the user passed a help flag as a positional argument (common on Unix shells
# e.g. `--help`) the param binder may leave it in the Stage list. Treat common
# help aliases as an explicit request for usage so we can exit before cloning
# or contacting Docker.
if ($script:Stages -contains '--help' -or $script:Stages -contains '-h' -or $script:Stages -contains '/?') {
    $Help = $true
    $script:Stages = $script:Stages | Where-Object { $_ -ne '--help' -and $_ -ne '-h' -and $_ -ne '/?' }
}
if ($Help) { Write-Verbose "Help requested; will show usage in main and exit." }

# Normalize common GNU-style `--stage` tokens that may have been passed as
# positional args (e.g. `--Stage linter` when invoking via `powershell -File`).
# Work against an array copy to avoid type issues when $script:Stages isn't a
# list-like object (can happen with odd parameter bindings).
$stagesArray = @($script:Stages)
$normalized = [System.Collections.Generic.List[string]]::new()
for ($i = 0; $i -lt $stagesArray.Count; $i++) {
    $val = $stagesArray[$i]
    if ($val -match '^--(?i:stage)=(.+)') {
        $normalized.Add($Matches[1])
        continue
    }
    if ($val -match '^--(?i:stage)$') {
        # If the token is `--Stage` and there's a following token, treat that as the value
        if ($i + 1 -lt $stagesArray.Count) {
            $next = $stagesArray[$i + 1]
            $normalized.Add($next)
            $i++
            continue
        }
        continue
    }
    if ($val -match '^--') {
        # Skip other unknown GNU-style long options passed as positional
        continue
    }
    $normalized.Add($val)
}
$script:Stages = $normalized

# Recover case where the user invoked like: script.ps1 --Stage linter
# which can result in the second token being bound to ProjectDir as a
# positional parameter. If ProjectDir appears to be one of the stage names
# (and is not an existing path), move it back into the stages list.
$projLeaf = Split-Path -Leaf $script:ProjectDir
if ($script:ALL_STAGES -contains $projLeaf) {
    if (-not (Test-Path $script:ProjectDir)) {
        $script:Stages.Add($projLeaf)
        $script:ProjectDir = (Get-Location).Path
    }
}

# ── Cleanup ──────────────────────────────────────────────────────────────────
function _cleanup {
    if ($script:CiScriptsTmp -and (Test-Path $script:CiScriptsTmp)) {
        log_debug "Removing temp ci-scripts clone: $($script:CiScriptsTmp)"
        Remove-Item -Recurse -Force $script:CiScriptsTmp -ErrorAction SilentlyContinue
    }

    if ($script:ProjectDir -and (Test-Path $script:ProjectDir)) {
        foreach ($f in $script:STRAY_REPORTS) {
            $target = Join-Path $script:ProjectDir $f
            if (Test-Path $target) {
                log_debug "Removing stray report: $target"
                Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue
            }
        }
    }
}

# Cleanup is invoked via try/finally around main at the bottom of the script.

# ── Banner / Header ───────────────────────────────────────────────────────────
function _banner {
    param([string]$Title)
    $width = 66
    $line  = "=" * $width
    Write-Host ""
    Write-Host $line               -ForegroundColor $script:ColCyan
    Write-Host ("  " + $Title.PadRight(62)) -ForegroundColor $script:ColCyan
    Write-Host $line               -ForegroundColor $script:ColCyan
}

function _header {
    param([string]$Name)
    Write-Host ""
    Write-Host "  Stage: $Name"     -ForegroundColor $script:ColBlue
    Write-Host ("  " + ("-" * 56)) -ForegroundColor $script:ColBlue
}

# ── Usage ─────────────────────────────────────────────────────────────────────
function Show-Usage {
        Write-Host @"

local-runner.ps1  -  Run DevSecOps CI stages locally via Docker

USAGE
    .\local-runner.ps1 [OPTIONS]

STAGES
    secret-scan   Gitleaks   - detect leaked secrets in source code
    sca           Trivy fs   - dependency vulnerability scan
    iac           Trivy cfg  - infrastructure misconfiguration scan
    linter        Language-aware linting (auto-detects JS/TS/Python/Java/Shell/Rust/Go)
    all           Run all five stages above  [default]

OPTIONS
    -Stage <name>          Stage to run; repeat for multiple (e.g. -Stage sca -Stage iac)
    -ProjectDir <path>     Project to scan  [default: current directory]
    -CiScriptsDir <path>   Path to a local ci-scripts clone  [default: auto-clone from GitHub]
    -Org <name>            GitHub org / registry namespace  [default: quilrbusiness]
    -LogLevel <level>      DEBUG | INFO | WARN | ERROR  [default: ERROR]
    -Language <lang>       Force linter language instead of auto-detect:
                                                 javascript | typescript | python | java | shell | rust | go
    -OutputDir <path>      Report output directory  [default: <project>\devsecops-precheck-reports]
    -Help                  Show this help and exit

EXAMPLES
    # Run only secret scan and SCA against a specific project
    .\local-runner.ps1 -Stage secret-scan -Stage sca -ProjectDir C:\projects\myapp

    # Lint a specific language with debug output
    .\local-runner.ps1 -Stage linter -Language python -LogLevel DEBUG

"@
        exit 0
}

# ── Convert Windows path to Docker-compatible path ───────────────────────────
# Converts C:\foo\bar  ->  /c/foo/bar  (for Docker Desktop on Windows)
function ConvertTo-DockerPath {
    param([string]$WinPath)
    $abs = [System.IO.Path]::GetFullPath($WinPath)
    # Replace backslashes, lowercase drive letter, strip colon
    $abs = $abs.Replace('\', '/')
    if ($abs -match '^([A-Za-z]):(.*)') {
        return '/' + $Matches[1].ToLower() + $Matches[2]
    }
    return $abs
}

# ── Resolve paths ─────────────────────────────────────────────────────────────
function _resolve_paths {
    $script:ProjectDir = [System.IO.Path]::GetFullPath($script:ProjectDir)
    $script:ProjectName = Split-Path -Leaf $script:ProjectDir
    if ($script:CiScriptsDir) {
        $script:CiScriptsDir = [System.IO.Path]::GetFullPath($script:CiScriptsDir)
    }
    if (-not $script:OutputDir) {
        $script:OutputDir = Join-Path $script:ProjectDir $script:OUTPUT_SUBDIR
    }
    $script:OutputDir = [System.IO.Path]::GetFullPath($script:OutputDir)

    log_debug "ProjectDir    : $($script:ProjectDir)"
    log_debug "ProjectName   : $($script:ProjectName)"
    log_debug "CiScriptsDir  : $($script:CiScriptsDir)"
    log_debug "OutputDir     : $($script:OutputDir)"
}

# Read a text file treating bytes as UTF-8 and return an array of lines.
function Read-TextFileUtf8Lines {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return @() }
    try {
        $bytes = [System.IO.File]::ReadAllBytes($Path)
        $text  = [System.Text.Encoding]::UTF8.GetString($bytes)
        $lines = $text -split "\n"
        return @($lines | ForEach-Object { $_.TrimEnd("`r") })
    } catch {
        return @(Get-Content $Path -ErrorAction SilentlyContinue)
    }
}

# ── Clone ci-scripts ──────────────────────────────────────────────────────────
function _clone_ci_scripts {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        log_fail "'git' is required to auto-clone ci-scripts. Install git or use -CiScriptsDir."
    }

    $sshUrl   = "git@github.com:$($script:Org)/$($script:CI_SCRIPTS_REPO).git"
    $httpsUrl = "https://github.com/$($script:Org)/$($script:CI_SCRIPTS_REPO).git"
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ci-scripts-" + [System.IO.Path]::GetRandomFileName().Replace('.',''))
    $script:CiScriptsTmp = $tmp
    $script:CiScriptsDir = $tmp

    log_info "Cloning $($script:Org)/$($script:CI_SCRIPTS_REPO)@$($script:CI_SCRIPTS_BRANCH) into $tmp"

    $cloned = $false
    foreach ($cloneUrl in @($sshUrl, $httpsUrl)) {
        $gitArgs = @("clone", "--branch", $script:CI_SCRIPTS_BRANCH, "--depth", "1", "--quiet", $cloneUrl, $tmp)
        log_debug "git $($gitArgs -join ' ')"
        $outFile = [System.IO.Path]::GetTempFileName()
        $errFile = [System.IO.Path]::GetTempFileName()
        $proc = Start-Process -FilePath git -ArgumentList $gitArgs -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $cloneRc = $proc.ExitCode
        Remove-Item $outFile, $errFile -ErrorAction SilentlyContinue
        if ($cloneRc -eq 0) { $cloned = $true; break }
        log_info "Clone via $cloneUrl failed, trying next protocol..."
        if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    }

    if (-not $cloned) {
        log_fail "Clone failed with both SSH and HTTPS (check -Org and network access)"
    }

    log_ok "ci-scripts ready at: $tmp"
}

# ── Print config ──────────────────────────────────────────────────────────────
function _print_config {
    _banner "DevSecOps Local Runner"
    Write-Host ""
    log_info "Project dir    : $($script:ProjectDir)"
    $cloneNote = if ($script:CiScriptsTmp) { "  (auto-cloned)" } else { "" }
    log_info "ci-scripts dir : $($script:CiScriptsDir)$cloneNote"
    log_info "Output dir     : $($script:OutputDir)"
    log_info "Stages         : $($script:Stages -join ' ')"
    log_info "Log level      : $($script:LogLevel)"
    Write-Host ""
}

# ── OS / dependency checks ────────────────────────────────────────────────────
function _check_os_compat {
    $missing = [System.Collections.Generic.List[string]]::new()

    function _command_exists {
        param([string]$Name)
        $cmd = Get-Command $Name -ErrorAction SilentlyContinue
        if ($cmd) { return $true }
        try {
            $where = & where.exe $Name 2>$null
            if ($where) { return $true }
        } catch { }
        return $false
    }

    if (-not (_command_exists 'docker')) {
        $missing.Add("Docker Desktop  ->  https://docs.docker.com/desktop/install/windows-install/")
    }
    if (-not (_command_exists 'git')) {
        $missing.Add("Git  ->  https://git-scm.com/download/win  (or: winget install Git.Git)")
    }
    if (-not (_command_exists 'jq')) {
        $missing.Add("jq   ->  winget install jqlang.jq          (or: choco install jq)")
    }
    if (-not (_command_exists 'yq')) {
        $missing.Add("yq   ->  winget install MikeFarah.yq        (or: choco install yq)")
    }

    if ($missing.Count -gt 0) {
        log_error "Missing Windows dependencies - install:"
        foreach ($hint in $missing) { log_error "  -> $hint" }
        exit 1
    }

    log_ok "Windows compatibility checks passed"
}

# ── Preflight ─────────────────────────────────────────────────────────────────
function preflight_check {
    _print_config
    _check_os_compat

    $outFile = [System.IO.Path]::GetTempFileName()
    $errFile = [System.IO.Path]::GetTempFileName()
    $proc = Start-Process -FilePath docker -ArgumentList 'info' -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    $dockerRc = $proc.ExitCode
    $dockerErr = if (Test-Path $errFile) { (Read-TextFileUtf8Lines $errFile -join "`n").Trim() } else { "" }
    Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
    if ($dockerRc -ne 0) {
        if ($dockerErr) { log_debug "docker info stderr: $dockerErr" }
        log_fail "Docker daemon is not running - start Docker Desktop and try again."
    }

    if (-not (Test-Path $script:ProjectDir)) {
        log_fail "Project directory not found: $($script:ProjectDir)"
    }

    $shellDir = Join-Path $script:CiScriptsDir "scripts\shell"
    if (-not (Test-Path $shellDir)) {
        log_fail "ci-scripts directory looks wrong: $($script:CiScriptsDir)"
    }

    New-Item -ItemType Directory -Force -Path $script:OutputDir | Out-Null
    log_ok "Preflight checks passed"
}

# ── GitHub context ────────────────────────────────────────────────────────────
function Build-GitHubContext {
    $sha    = & git -C $script:ProjectDir rev-parse HEAD 2>$null
    if (-not $sha) { $sha = "local-$(Get-Date -UFormat '%s')" }

    $branch = & git -C $script:ProjectDir rev-parse --abbrev-ref HEAD 2>$null
    if (-not $branch) { $branch = "local" }

    $repo   = Split-Path -Leaf $script:ProjectDir
    $runId  = "local-$(Get-Date -UFormat '%s')"

    $ctx = [ordered]@{
        workspace        = "$($script:WORKSPACE_MOUNT)/$($script:ProjectName)"
        sha              = $sha
        ref_name         = $branch
        repository       = "$($script:Org)/$repo"
        repository_owner = $script:Org
        run_id           = $runId
        event_name       = "push"
        server_url       = "https://github.com"
    }

    return ($ctx | ConvertTo-Json -Compress)
}

# ── run_in_docker ─────────────────────────────────────────────────────────────
function run_in_docker {
    param(
        [string]   $StageName,
        [string]   $Image,
        [string]   $Script,
        [string[]] $ExtraEnvs = @()
    )

    log_debug "-- run_in_docker --"
    log_debug "StageName : $StageName"
    log_debug "Image     : $Image"
    log_debug "Script    : $Script"

    $ctx = Build-GitHubContext
    log_debug "ctx : $ctx"
    # Set via process env so Docker inherits it by name, avoiding cmd.exe quote mangling
    $env:GITHUB_CONTEXT = $ctx

    $stageReport = Join-Path $script:OutputDir $StageName
    New-Item -ItemType Directory -Force -Path $stageReport | Out-Null

    $containerWorkspace = "$($script:WORKSPACE_MOUNT)/$($script:ProjectName)"
    $projMount     = ConvertTo-DockerPath $script:ProjectDir
    $ciMount       = ConvertTo-DockerPath $script:CiScriptsDir
    $reportMount   = ConvertTo-DockerPath $stageReport
    $containerName = "devsecops-$($StageName -replace '/','-')-$PID"

    $dockerArgs = @(
        "run", "--rm",
        "--name", $containerName,
        "--platform", "linux/amd64",
        "-v", "${projMount}:$containerWorkspace",
        "-v", "${ciMount}:${containerWorkspace}/ci-scripts",
        "-v", "${reportMount}:${containerWorkspace}/artifacts",
        "-e", "GITHUB_WORKSPACE=$containerWorkspace",
        "-e", "LOG_LEVEL=$($script:LogLevel)",
        "-e", "GITHUB_CONTEXT"
    )

    foreach ($env in $ExtraEnvs) {
        $dockerArgs += @("-e", $env)
    }

    # Run a small bootstrap inside the container to normalize line endings
    # (convert CRLF -> LF) for mounted scripts coming from Windows hosts, then
    # exec the script. This prevents `/opt/.../script.sh: $'\r': command not found`.
    # Use a double-quoted PowerShell string so $Script is expanded, with
    # single-quotes inside the shell command to avoid nested double-quote escape
    # issues in PowerShell. Normalize the main script and all .sh files under
    # the mounted ci-scripts directory to remove CRLFs before executing.
    $ciScriptsMount = "${containerWorkspace}/ci-scripts"
    # Ensure git treats the mounted workspace as safe, normalize CRLFs in
    # scripts under the ci-scripts folder, then execute the stage script.
    $runCmd = "git config --global --add safe.directory $containerWorkspace; if [ -f '$Script' ]; then sed -i 's/\r$//' '$Script'; fi; if [ -d '$ciScriptsMount' ]; then find '$ciScriptsMount' -type f -name '*.sh' -print0 | xargs -0 -r sed -i 's/\r$//' ; fi; exec bash '$Script'"
    log_debug "Container run command: $runCmd"
    $dockerArgs += @($Image, "bash", "-lc", $runCmd)

    log_debug "Docker cmd: docker $($dockerArgs -join ' ')"

    $t0 = Get-Date
    $rc = 0

    if ($script:LogLevel -eq "DEBUG") {
        # Build a single, safely-quoted command string for cmd.exe so complex
        # JSON env values and other args with spaces are preserved correctly.
        $argQuoted = @()
        foreach ($a in $dockerArgs) {
            if ($a -match '\s' -or $a -match '"') {
                $escaped = $a.Replace('"', '\\"')
                $argQuoted += '"' + $escaped + '"'
            } else {
                $argQuoted += $a
            }
        }
        $cmdLine = 'docker ' + ($argQuoted -join ' ')

        # Start docker via cmd.exe and redirect output to temp files, then
        # poll the files and print new lines as they arrive to stream logs
        # without using async event handlers (which can be unstable on PS5.1).
        $outFile = [System.IO.Path]::GetTempFileName()
        $errFile = [System.IO.Path]::GetTempFileName()
        $proc = Start-Process -FilePath cmd.exe -ArgumentList '/c', $cmdLine -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile

        $lastOut = 0
        $lastErr = 0
        while (-not $proc.HasExited) {
            if (Test-Path $outFile) {
                $lines = @(Read-TextFileUtf8Lines $outFile)
                for ($i = $lastOut; $i -lt $lines.Count; $i++) { Write-Host (Normalize-ConsoleString $lines[$i]) }
                $lastOut = $lines.Count
            }
            if (Test-Path $errFile) {
                $elines = @(Read-TextFileUtf8Lines $errFile)
                for ($i = $lastErr; $i -lt $elines.Count; $i++) { Write-Host $elines[$i] }
                $lastErr = $elines.Count
            }
            Start-Sleep -Milliseconds 200
        }

        # Print any remaining lines after process exit
        if (Test-Path $outFile) {
            $lines = @(Read-TextFileUtf8Lines $outFile)
            for ($i = $lastOut; $i -lt $lines.Count; $i++) { Write-Host (Normalize-ConsoleString $lines[$i]) }
        }
        if (Test-Path $errFile) {
            $elines = @(Read-TextFileUtf8Lines $errFile)
            for ($i = $lastErr; $i -lt $elines.Count; $i++) { Write-Host $elines[$i] }
        }

        # Ensure process state is finalized before reading exit code.
        $proc.WaitForExit()
        $proc.Refresh()
        $rc = $proc.ExitCode
        Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
        Write-Host ""
    } else {
        # Non-DEBUG: run docker via cmd.exe and capture output to temp files
        # to avoid PowerShell NativeCommandError when docker/contained tools
        # emit stderr; we discard the output but preserve the exit code.
        $outFile = [System.IO.Path]::GetTempFileName()
        $errFile = [System.IO.Path]::GetTempFileName()
        $argQuoted = @()
        foreach ($a in $dockerArgs) {
            if ($a -match '\s' -or $a -match '"') {
                $escaped = $a.Replace('"', '\\"')
                $argQuoted += '"' + $escaped + '"'
            } else {
                $argQuoted += $a
            }
        }
        $cmdLine = 'docker ' + ($argQuoted -join ' ')
        $proc = Start-Process -FilePath cmd.exe -ArgumentList '/c', $cmdLine -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $rc = $proc.ExitCode
        Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
        Write-Host ""
    }

    $rc = Normalize-ExitCode $rc
    Remove-Item Env:\GITHUB_CONTEXT -ErrorAction SilentlyContinue

    $elapsed = [int]((Get-Date) - $t0).TotalSeconds

    if ($rc -eq 0) {
        log_ok "Stage '$StageName' passed  [${elapsed}s]"
    } else {
        # If the container exited non-zero, inspect generated reports (e.g. Trivy
        # JSON) to determine if there were actually any findings.
        $overrideToZero = $false
        try {
            if ($StageName -in @('iac','sca')) {
                $trivyHostJson = Join-Path $script:ProjectDir 'trivy-report.json'
                if (Test-Path $trivyHostJson) {
                    $findingsCount = $null

                    # Prefer jq for parity with shell scripts and better resilience
                    # across Trivy JSON shape changes.
                    if (Get-Command jq -ErrorAction SilentlyContinue) {
                        try {
                            $jqExpr = if ($StageName -eq 'iac') {
                                '[.Results[]?.Misconfigurations[]?, .Results[]?.Secrets[]?] | length'
                            } else {
                                '[.Results[]?.Vulnerabilities[]?] | length'
                            }
                            $jqOut = & jq -r $jqExpr $trivyHostJson 2>$null
                            if ($LASTEXITCODE -eq 0 -and $jqOut) {
                                $n = 0
                                if ([int]::TryParse("$jqOut".Trim(), [ref]$n)) {
                                    $findingsCount = $n
                                }
                            }
                        } catch {
                            # Fall through to PowerShell JSON parsing.
                        }
                    }

                    # Fallback parser when jq is unavailable or returned nothing.
                    if ($null -eq $findingsCount) {
                        $raw = Get-Content $trivyHostJson -ErrorAction SilentlyContinue -Raw
                        if ($raw) {
                            try {
                                $j = $raw | ConvertFrom-Json -ErrorAction Stop
                                $found = @()
                                foreach ($r in @($j.Results)) {
                                    if ($StageName -eq 'iac') {
                                        if ($r.Misconfigurations) { foreach ($m in $r.Misconfigurations) { $found += $m } }
                                        if ($r.Secrets) { foreach ($s in $r.Secrets) { $found += $s } }
                                    } else {
                                        if ($r.Vulnerabilities) { foreach ($v in $r.Vulnerabilities) { $found += $v } }
                                    }
                                }
                                $findingsCount = ($found | Measure-Object).Count
                            } catch {
                                # Keep null if parsing fails.
                            }
                        }
                    }

                    if ($null -ne $findingsCount -and $findingsCount -eq 0) {
                        $overrideToZero = $true
                    }
                }
            }
        } catch { }

        if ($overrideToZero) {
            log_ok "Stage '$StageName' passed (no findings detected in report)  [${elapsed}s]"
            $rc = 0
        } else {
            log_warn "Stage '$StageName' finished with findings  [exit=$rc, ${elapsed}s]"
            log_warn "Review $StageName reports in: $stageReport"
        }
    }

    return $rc
}

# Prefer a local image when available; fall back to the requested fully-qualified
# image. This avoids pulling from GHCR (which may be private) when a suitable
# image is already present locally. Returns the chosen image string.
function Resolve-Image {
    param([string]$Requested)

    log_debug "Resolve-Image: requested: $Requested"

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { return $Requested }

    # Candidate list: exact requested, org/repo:tag, repo:tag (local), org/repo:tag without registry
    $candidates = @()
    $candidates += $Requested

    $parts = $Requested -split '/'
    $last = $parts[-1]
    if ($last) {
        $candidates += $last
        if ($script:Org) { $candidates += "$($script:Org)/$last" }
    }

    foreach ($cand in $candidates) {
        try {
            & docker image inspect $cand > $null 2>$null
            if ($LASTEXITCODE -eq 0) {
                log_debug "Resolve-Image: using local image: $cand"
                return $cand
            }
        } catch {
            # ignore
        }
    }

    # Nothing local — return original request but give a helpful hint if pull will fail
    log_debug "Resolve-Image: no local image found for candidates: $($candidates -join ', ')"
    return $Requested
}

# ── Secret Scan ───────────────────────────────────────────────────────────────
function run_secret_scan {
    log_debug "-- run_secret_scan --"
    _header "Secret Scan  (Gitleaks)"

    $image = "$($script:REGISTRY)/$($script:IMG_SECRET_SCAN)"
    $image = Resolve-Image $image
    log_debug "image : $image"

    return (run_in_docker `
        -StageName "secret-scan" `
        -Image     $image `
        -Script    "$($script:WORKSPACE_MOUNT)/$($script:ProjectName)/ci-scripts/scripts/shell/secret_scan/secret_scan.sh")
}

# ── SCA ───────────────────────────────────────────────────────────────────────
function run_sca {
    log_debug "-- run_sca --"
    _header "SCA  (Trivy - dependency vulns)"

    $image = "$($script:REGISTRY)/$($script:IMG_SCA)"
    $image = Resolve-Image $image
    log_debug "image : $image"

    return (run_in_docker `
        -StageName "sca" `
        -Image     $image `
        -Script    "$($script:WORKSPACE_MOUNT)/$($script:ProjectName)/ci-scripts/scripts/shell/sca/sca.sh" `
        -ExtraEnvs @("TRIVY_CACHE_DIR=/tmp/trivy-cache"))
}

# ── IAC ───────────────────────────────────────────────────────────────────────
function run_iac {
    log_debug "-- run_iac --"
    _header "IAC  (Trivy - infra misconfigs)"

    $image = "$($script:REGISTRY)/$($script:IMG_IAC)"
    $image = Resolve-Image $image
    log_debug "image : $image"

    return (run_in_docker `
        -StageName "iac" `
        -Image     $image `
        -Script    "$($script:WORKSPACE_MOUNT)/$($script:ProjectName)/ci-scripts/scripts/shell/iac/iac.sh" `
        -ExtraEnvs @("TRIVY_CACHE_DIR=/tmp/trivy-cache"))
}

# ── Linter helpers ────────────────────────────────────────────────────────────
function Invoke-Yq {
    param(
        [string] $Query,
        [string] $YamlPath
    )

    $outFile = [System.IO.Path]::GetTempFileName()
    $errFile = [System.IO.Path]::GetTempFileName()

    Start-Process -FilePath yq -ArgumentList '-r', $Query, $YamlPath -NoNewWindow -Wait -RedirectStandardOutput $outFile -RedirectStandardError $errFile

    $out = @()
    if (Test-Path $outFile) {
        $out = @(Read-TextFileUtf8Lines $outFile | ForEach-Object { "$_".Trim() })
    }
    Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
    return $out
}

function Get-FirstMeaningfulYqValue {
    param([string[]]$Lines)
    foreach ($line in @($Lines)) {
        if ($null -eq $line) { continue }
        $trimmed = "$line".Trim()
        if (-not $trimmed) { continue }
        if ($trimmed -eq "null") { continue }
        return $trimmed
    }
    return $null
}

function _image_for_language {
    param([string]$Lang)

    $yaml = Join-Path $script:CiScriptsDir "config\yaml\linter\linter.yaml"
        $imgQuery = '.language.' + $Lang + '.image'
        $tagQuery = '.language.' + $Lang + '."image-tag"'
        $imgLines  = Invoke-Yq $imgQuery $yaml
        $tagLines  = Invoke-Yq $tagQuery $yaml
        $img = Get-FirstMeaningfulYqValue $imgLines
        $tag = Get-FirstMeaningfulYqValue $tagLines

    if (-not $img -or -not $tag) {
        log_error "No image entry for language '$Lang' in linter.yaml"
        return $null
    }

    return "$($script:REGISTRY)/${img}:${tag}"
}

function _get_linter_matrix {
    $yaml = Join-Path $script:CiScriptsDir "config\yaml\linter\linter.yaml"

    # Build prune dirs
    $excludeDirs = & yq -r '.exclude_dirs[]?' $yaml 2>$null
    $pruneDirs   = @()
    if ($excludeDirs) {
        foreach ($d in $excludeDirs) {
            if ($d) { $pruneDirs += Join-Path $script:ProjectDir $d }
        }
    }

    $include  = [System.Collections.Generic.List[hashtable]]::new()
    $allLangs = & yq -r '.language | keys[]' $yaml 2>$null

    foreach ($lang in $allLangs) {
        if (-not $lang) { continue }

            $patternsQuery = '.language.' + $lang + '.files[]?'
            $patternsArg = $patternsQuery
            $patterns = Invoke-Yq $patternsArg $yaml
        $matched  = $false

        foreach ($pattern in $patterns) {
            if (-not $pattern) { continue }

            # Search for matching files, respecting exclude dirs
            $hit = Get-ChildItem -Path $script:ProjectDir -Filter $pattern -Recurse -ErrorAction SilentlyContinue |
                   Where-Object {
                       $filePath = $_.FullName
                       -not ($pruneDirs | Where-Object { $filePath.StartsWith($_) })
                   } |
                   Select-Object -First 1

            if ($hit) {
                $matched = $true
                log_debug "[$lang] detected via: $($hit.FullName)"
                break
            }
        }

        if ($matched) {
                $imgQuery = '.language["' + $lang + '"].image'
                $tagQuery = '.language["' + $lang + '"]["image-tag"]'
                $imgLines = Invoke-Yq $imgQuery $yaml
                $tagLines = Invoke-Yq $tagQuery $yaml
                $img = Get-FirstMeaningfulYqValue $imgLines
                $tag = Get-FirstMeaningfulYqValue $tagLines
                $include.Add(@{ language = $lang; img = $img; tag = $tag })
        }
    }

    if ($include.Count -eq 0) { return $null }

    return @{ include = $include.ToArray() }
}

# ── Linter ────────────────────────────────────────────────────────────────────
function run_linter {
    _header "Linter  (language-aware)"

    if (-not (Get-Command yq -ErrorAction SilentlyContinue)) {
        log_fail "Linter stage requires 'yq' - install it (winget install MikeFarah.yq / choco install yq)."
    }

    $langs      = [System.Collections.Generic.List[string]]::new()
    $langImages = @{}

    log_debug "LinterLanguage : $($script:LinterLanguage)"

    if ($script:LinterLanguage) {
        $langs.Add($script:LinterLanguage)

        $image = _image_for_language $script:LinterLanguage
        if (-not $image) {
            log_error "Cannot resolve image for '$($script:LinterLanguage)'"
            return 1
        }
        $langImages[$script:LinterLanguage] = $image
        log_debug "Language (forced): $($script:LinterLanguage)  ->  $image"

    } else {
        log_debug "Detecting languages via linter matrix ..."

        $matrix = _get_linter_matrix

        if (-not $matrix) {
            log_warn "No supported languages detected - skipping linter stage"
            return 0
        }

        foreach ($entry in $matrix.include) {
            $langs.Add($entry.language)
            # Resolve directly from YAML to avoid matrix/object shape issues on PS Windows.
            $langImages[$entry.language] = _image_for_language $entry.language
        }
    }

    if ($langs.Count -eq 0) {
        log_warn "No supported languages detected - skipping linter stage"
        return 0
    }

    log_info "Languages: $($langs -join ' ')"

    $overallRc = 0

    foreach ($lang in $langs) {
        if ($lang -eq "rust" -and -not $env:CLONE_TOKEN) {
            log_error "CLONE_TOKEN is required for the Rust linter but is not set."
            log_error "  -> set CLONE_TOKEN in your environment and retry."
            return 1
        }

        $image = $langImages[$lang]
        if (-not $image -or -not "$image".Trim()) {
            $image = _image_for_language $lang
        }
        $image = Resolve-Image $image
        if (-not $image) {
            log_warn "No linter image for '$lang' - skipping"
            continue
        }

        $ctx         = Build-GitHubContext
        $env:GITHUB_CONTEXT = $ctx
        $stageReport = Join-Path $script:OutputDir "linter-$lang"
        New-Item -ItemType Directory -Force -Path $stageReport | Out-Null

        $containerWorkspace = "$($script:WORKSPACE_MOUNT)/$($script:ProjectName)"
        $projMount   = ConvertTo-DockerPath $script:ProjectDir
        $ciMount     = ConvertTo-DockerPath $script:CiScriptsDir
        $reportMount = ConvertTo-DockerPath $stageReport
        $contName    = "devsecops-linter-${lang}-$PID"

        Write-Host ""
        log_debug "  [$lang]  Image : $image"
        log_debug "  [$lang]  Output: $stageReport"

        $linterArgs = @(
            "run", "--rm",
            "--name", $contName,
            "--platform", "linux/amd64",
            "-v", "${projMount}:$containerWorkspace",
            "-v", "${ciMount}:${containerWorkspace}/ci-scripts",
            "-v", "${reportMount}:${containerWorkspace}/artifacts",
            "-e", "GITHUB_WORKSPACE=$containerWorkspace",
            "-e", "GIT_BASE_URL=github.com/$($script:Org)",
            "-e", "CLONE_TOKEN=$($env:CLONE_TOKEN)",
            "-e", "MAVEN_CONFIG=/tmp/.m2",
            "-e", "LOG_LEVEL=$($script:LogLevel)",
            "-e", "GITHUB_CONTEXT"
        )

        $ciScriptsMount = "${containerWorkspace}/ci-scripts"
        $linterScript = "${containerWorkspace}/ci-scripts/scripts/shell/linter/linter.sh"
        $runCmd = "git config --global --add safe.directory $($script:WORKSPACE_MOUNT); if [ -f '$linterScript' ]; then sed -i 's/\r$//' '$linterScript'; fi; if [ -d '$ciScriptsMount' ]; then find '$ciScriptsMount' -type f -name '*.sh' -print0 | xargs -0 -r sed -i 's/\r$//' ; fi; if [ -z `"$HOME`" ]; then HOME=`"/home/github-ci`"; fi; if [ -f `"/etc/profile`" ]; then . `"/etc/profile`"; fi; if [ -f `"$HOME/.profile`" ]; then . `"$HOME/.profile`"; fi; if [ -f `"$HOME/.bashrc`" ]; then . `"$HOME/.bashrc`"; fi; if [ -f `"$HOME/.cargo/env`" ]; then . `"$HOME/.cargo/env`"; fi; export PATH=`"$HOME/.cargo/bin:/usr/local/cargo/bin:`$PATH`"; exec bash '$linterScript' '$lang'"
        log_debug "  [$lang]  Container run command: $runCmd"
        $linterArgs += @(
            $image,
            "bash", "-lc", $runCmd
        )

        $t0 = Get-Date
        $rc = 0

        if ($script:LogLevel -eq "DEBUG") {
            $prevErrorAction = $ErrorActionPreference
            try {
                # Native tools often emit warnings on stderr; do not treat them as
                # terminating errors while still streaming output in DEBUG mode.
                $ErrorActionPreference = "Continue"
                & docker @linterArgs 2>&1 | ForEach-Object {
                    Write-Host (Normalize-ConsoleString "$_")
                }
                $rc = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $prevErrorAction
            }
        } else {
            $outFile = [System.IO.Path]::GetTempFileName()
            $errFile = [System.IO.Path]::GetTempFileName()
            $argQuoted = @()
            foreach ($a in $linterArgs) {
                if ($a -match '\s' -or $a -match '"') {
                    $escaped = $a.Replace('"', '\\"')
                    $argQuoted += '"' + $escaped + '"'
                } else {
                    $argQuoted += $a
                }
            }
            $cmdLine = 'docker ' + ($argQuoted -join ' ')
            $proc = Start-Process -FilePath cmd.exe -ArgumentList '/c', $cmdLine -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
            $rc = $proc.ExitCode
            Remove-Item $outFile,$errFile -ErrorAction SilentlyContinue
        }

        Remove-Item Env:\GITHUB_CONTEXT -ErrorAction SilentlyContinue
        $elapsed = [int]((Get-Date) - $t0).TotalSeconds

        if ($rc -eq 0) {
            log_ok "Linter '$lang' passed  [${elapsed}s]"
        } else {
            log_warn "Linter '$lang' finished with findings  [exit=$rc, ${elapsed}s]"
                log_warn "Reports in: $stageReport"
            $overallRc = $rc
        }
    }

    return $overallRc
}

# ── Summary ───────────────────────────────────────────────────────────────────
function print_summary {
    _banner "Summary"
    Write-Host ""

    $anyFail    = $false
    $colStage   = 20
    $colStatus  = 12
    $colExit    = 6
    $colReport  = 45
    $width      = $colStage + $colStatus + $colExit + $colReport + 8
    $divider    = "-" * $width

    Write-Host ("  " + "STAGE".PadRight($colStage) + "  " +
                "STATUS".PadRight($colStatus)       + "  " +
                "EXIT".PadRight($colExit)            + "  " +
                "REPORT PATH".PadRight($colReport)) -ForegroundColor $script:ColCyan
    Write-Host ("  " + $divider) -ForegroundColor $script:ColCyan

    foreach ($stage in $script:StageResults.Keys) {
        $rc         = $script:StageResults[$stage]
        $reportPath = Join-Path $script:OutputDir $stage

        if ($stage -eq "linter") {
            $linterDirs = Get-ChildItem -Path $script:OutputDir -Directory -Filter "linter-*" -ErrorAction SilentlyContinue
            if ($linterDirs) {
                $reportPath = ($linterDirs | ForEach-Object { $_.FullName }) -join ", "
            }
        }

        $statusTxt = if ($rc -eq 0) { "PASSED" } else { "FINDINGS" }
        $color     = if ($rc -eq 0) { $script:ColGreen } else { $script:ColRed }

        Write-Host ("  " + $stage.PadRight($colStage) + "  " +
                    $statusTxt.PadRight($colStatus)    + "  " +
                    "$rc".PadRight($colExit)            + "  " +
                    $reportPath.PadRight($colReport)) -ForegroundColor $color

        if ($rc -ne 0) { $anyFail = $true }
    }

    Write-Host ("  " + $divider) -ForegroundColor $script:ColCyan
    Write-Host ""

    if (-not $anyFail) {
        log_ok "All stages passed with no findings."
    } else {
        log_warn "One or more stages reported findings - review the reports above."
    }
}

function Normalize-ExitCode {
    param($Value)

    if ($null -eq $Value) { return 1 }

    if ($Value -is [System.Array]) {
        foreach ($item in ($Value | Select-Object -Reverse)) {
            $parsed = Normalize-ExitCode $item
            if ($parsed -ne $null) { return $parsed }
        }
        return 1
    }

    if ($Value -is [int]) { return $Value }

    $text = "$Value".Trim()
    if (-not $text) { return 1 }

    $m = [regex]::Match($text, '^-?\d+')
    if ($m.Success) {
        try { return [int]$m.Value } catch { }
    }

    return 1
}

# ── Main ──────────────────────────────────────────────────────────────────────
function main {
    if ($Help) { Show-Usage }

    # Resolve LogLevel from parameter (parse_args equivalent already done via params)
    $script:LogLevel = $LogLevel.ToUpper()

    # Ensure $script:Stages is a list-like object so .Count and .Add work reliably
    if (-not ($script:Stages -is [System.Collections.ICollection] -and -not ($script:Stages -is [string]))) {
        $tmpList = [System.Collections.Generic.List[string]]::new()
        foreach ($s in @($script:Stages)) { if ($s) { $tmpList.Add($s) } }
        $script:Stages = $tmpList
    }

    $numStages = @($script:Stages).Count
    log_debug "# of Stages : $numStages"

    if ($numStages -eq 0) { $script:Stages.Add("all") }

    if ($script:Stages -contains "all") {
        $script:Stages = [System.Collections.Generic.List[string]]::new()
        foreach ($s in $script:ALL_STAGES) { $script:Stages.Add($s) }
    }

    $numStages = @($script:Stages).Count
    log_debug "# of Stages : $numStages"
    log_debug "Stages      : $($script:Stages -join ', ')"

    _resolve_paths

    if (-not $script:CiScriptsDir) { _clone_ci_scripts }

    preflight_check

    foreach ($stage in $script:Stages) {
        $rc = 0
        switch ($stage) {
            "secret-scan" { $rc = run_secret_scan }
            "sca"         { $rc = run_sca         }
            "iac"         { $rc = run_iac          }
            "linter"      { $rc = run_linter       }
            default {
                log_error "Unknown stage: '$stage'"
                log_error "Valid stages: $($script:ALL_STAGES -join ', ')  (or: all)"
                exit 1
            }
        }
        $script:StageResults[$stage] = Normalize-ExitCode $rc
    }

    print_summary
}

try {
    main
} finally {
    _cleanup
}
