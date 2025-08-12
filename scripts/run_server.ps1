#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

uvicorn src.chapter9.server_fastapi:app --reload
