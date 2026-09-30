# E8b: the memorisation perturbation (A2 arm) on the three local 7B models,
# requested by NeurIPS reviewer BJWk as "the obvious missing column".
# Same prompt template, temperature 0.8 and max_tokens 1200 as the main local
# corpus; perturbed tasks from benchmark/tasks_e8.yaml; one file per model so the
# main corpus (outcomes_*.jsonl) is never touched. Sequential on purpose: the
# models share one CPU.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
$tasks = @("e8_a2_count", "e8_a2_sum", "e8_a2_mean", "e8_a2_rr")
foreach ($m in "qwen2.5-coder:7b", "deepseek-coder:6.7b", "codellama:7b") {
    $safe = $m.Replace(":", "_")
    $args = @("generators\generation_runner.py", "--model", $m, "--provider", "openai",
              "--base-url", "http://localhost:11434/v1", "--samples", "10",
              "--max-tokens", "1200", "--sample-major", "--resume",
              "--tasks-file", "benchmark\tasks_e8.yaml",
              "--out", "results\raw\gen_e8b_$safe.jsonl", "--tasks") + $tasks
    & .\.venv\Scripts\python.exe @args *>> "results\e8b_generation.log"
}
& .\.venv\Scripts\python.exe analysis\aggregate.py --raw-glob "results/raw/gen_e8b_*.jsonl" `
    --tasks-file benchmark\tasks_e8.yaml --out results\processed\e8b_outcomes.jsonl *>> "results\e8b_generation.log"
& .\.venv\Scripts\python.exe analysis\e8_report.py --outcomes results\processed\e8b_outcomes.jsonl *>> "results\e8b_report.log"
"=== done $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 "results\e8b_generation.log"
