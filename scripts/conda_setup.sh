#!/usr/bin/env bash
set -euo pipefail
conda create -n langchain-book python=3.11 -y
conda activate langchain-book || source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate langchain-book
pip install -r requirements.txt
echo "Env ready: 'conda activate langchain-book'"
