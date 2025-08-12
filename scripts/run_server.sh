#!/usr/bin/env bash
set -euo pipefail
uvicorn src.chapter9.server_fastapi:app --reload
