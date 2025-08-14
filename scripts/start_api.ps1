param(
  [int]$Port = 8010
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve repo root as parent of this scripts directory
$repo = Split-Path -LiteralPath $PSScriptRoot -Parent
Set-Location -LiteralPath $repo

# Ensure venv exists
if (-not (Test-Path "$repo\.venv\Scripts\python.exe")) {
  throw "Virtual env not found at $repo\.venv. Run scripts/venv_setup.ps1 first."
}

$env:USER_AGENT = 'generative-ai-with-langchain-akse/0.1'

# Use venv python to launch uvicorn to avoid PATH issues with spaces
& "$repo\.venv\Scripts\python.exe" -m uvicorn src.chapter9.server_fastapi:app --host 127.0.0.1 --port $Port

