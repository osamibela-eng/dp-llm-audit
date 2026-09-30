# E12: extend the hosted-model arm from one model to four.
# Same prompt, temperature, max_tokens, sample count and ordering as E5
# (docs/E5_FRONTIER_SETUP.md). Each model gets its own raw file and log;
# nothing here touches the local corpus (outcomes_*.jsonl).
#   powershell -File run_e12_models.ps1 -Model gemma-4-31b-it
param([Parameter(Mandatory = $true)][string]$Model)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
$safe = $Model.Replace(":", "_").Replace("/", "_")
$log = "results\e12_generation_$safe.log"
"=== $Model  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
& .\.venv\Scripts\python.exe generators\generation_runner.py --model $Model --provider openai `
    --base-url "https://generativelanguage.googleapis.com/v1beta/openai" `
    --api-key-env GEMINI_API_KEY --samples 10 --max-tokens 6000 `
    --sample-major --resume --out "results\raw\gen_e12_$safe.jsonl" *>> $log
"=== done $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
