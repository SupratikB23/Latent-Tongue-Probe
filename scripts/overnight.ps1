# Unattended run on the RTX 3070 box, ~3.5-4 h. From the repo root (venv activation not needed):
#   powershell -ExecutionPolicy Bypass -File scripts\overnight.ps1
#
#   1. main     configs/amended.yaml  BLOOM-560m (float32) + BLOOM-1b7 (bfloat16), 5 seeds -> results_amended/
#   2. diagnose baseline table for the caches used in step 1                    -> logs/<stamp>_2_diagnose.out
#   3. backup   configs/backup.yaml   side_aunt concept, same models and seeds    -> results_backup/
#
# A failing step is logged and the next one still runs. The machine is kept awake while the script runs.
# Progress: Get-Content logs\<stamp>_1_main.err -Tail 5 -Wait     Summary: logs\<stamp>_summary.log

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$summary = "logs\${stamp}_summary.log"

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: block sleep for this process only; released automatically on exit
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

function Log($msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Host $line
    Add-Content -Path $summary -Value $line -Encoding UTF8
}

function Step($name, [string[]]$pyArgs) {
    $t0 = Get-Date
    Log "START $name :: python $($pyArgs -join ' ')"
    # Windows PowerShell 5.1 joins ArgumentList with spaces without quoting, so quote anything containing whitespace
    $argList = (@("-X", "faulthandler") + $pyArgs) | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "logs\${stamp}_${name}.out" -RedirectStandardError "logs\${stamp}_${name}.err"
    Log ("END   $name :: exit {0}, {1:N0} min" -f $p.ExitCode, ((Get-Date) - $t0).TotalMinutes)
}

& $py -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" 2>&1 | ForEach-Object { Log $_ }

Step "1_main"     @("src/run_all.py", "--config", "configs/amended.yaml")
Step "2_diagnose" @("src/diagnose_baseline.py")
Step "3_backup"   @("src/run_all.py", "--config", "configs/backup.yaml")

foreach ($v in "results_amended\verdict.md", "results_backup\verdict.md") {
    if (Test-Path $v) { Log "----- $v -----"; Get-Content $v -Encoding UTF8 | ForEach-Object { Add-Content $summary $_ -Encoding UTF8 } }
    else { Log "MISSING $v (see the .err log of that step)" }
}
Log "ALL DONE"
