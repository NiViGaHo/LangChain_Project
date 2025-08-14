param(
  [int]$Port = 8010,
  [string]$Message = 'What is LCEL?'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$headers = @{ 'Content-Type' = 'application/json' }
$body = @{ message = $Message } | ConvertTo-Json
$uri = "http://127.0.0.1:$Port/chat"

$res = Invoke-RestMethod -Method Post -Uri $uri -Headers $headers -Body $body
$res | ConvertTo-Json -Compress

