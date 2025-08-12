#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-Not (Test-Path .venv)) {
  python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt

Write-Host "Env ready. Activate next time with: . .\.venv\Scripts\Activate.ps1"
