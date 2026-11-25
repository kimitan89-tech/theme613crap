# theme613crap launcher: runs the title for normal play (no scripted presses, no capture).
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Console = "series-s",
    [string]$Cache = "$env:LOCALAPPDATA\victory\cache",
    [string]$Data = "$env:LOCALAPPDATA\victory\data",
    [string]$Exe = "$PSScriptRoot\..\target\release\victory.exe"
)
foreach ($v in "VICTORY_KEYS", "VICTORY_CAPTURE", "VICTORY_CAPTURE_FRAMES", "VICTORY_DUMP_TARGETS") {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force $Cache, $Data | Out-Null
$env:VICTORY_CACHE_DIR = $Cache
$env:VICTORY_DATA_DIR = $Data
& $Exe run $Game --execute --console $Console 2>&1 | Out-File "$env:TEMP\theme613crap.log" -Encoding utf8
Write-Host "Exited with $LASTEXITCODE; log: $env:TEMP\theme613crap.log"
