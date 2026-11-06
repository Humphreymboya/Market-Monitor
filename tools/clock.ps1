# Clock for the monitor. GitHub has not served a single event:schedule run for this repo,
# so this machine supplies the trigger and GitHub's runner still does the work. One brain,
# one sender: the watchlist logic and state/positions.json live only on the runner, and the
# workflow's concurrency group means overlapping ticks queue instead of double-alerting.
#
# Registered as a scheduled task repeating every 5 minutes:
#   schtasks /create /tn MarketMonitor-Tick /sc minute /mo 5 /f `
#     /tr "powershell -NoProfile -ExecutionPolicy Bypass -File <this file>"
# Remove it with:  schtasks /delete /tn MarketMonitor-Tick /f
#
# Nothing here reads the ntfy topic or a token: gh uses the credentials already in the
# Windows keyring, and the only thing sent to GitHub is "start a pass".

$ErrorActionPreference = "Continue"
$repo = "Humphreymboya/Market-Monitor"
$dir  = Join-Path $env:LOCALAPPDATA "MarketMonitor"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$log = Join-Path $dir "clock.log"

$out = & gh workflow run live.yml --repo $repo 2>&1
$code = $LASTEXITCODE

# The log proves the clock is ticking; it is capped because it runs unattended forever.
if ((Test-Path $log) -and ((Get-Item $log).Length -gt 1MB)) { Remove-Item $log -Force }
$stamp = (Get-Date).ToUniversalTime().ToString("u")
Add-Content $log ("{0} dispatch {1} {2}" -f $stamp, $(if ($code -eq 0) { "ok" } else { "exit=$code" }), ($out -join " "))
