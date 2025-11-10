#!/bin/bash

args=("$@")
cur_dir=$(dirname "$0")

# Test if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv could not be found, installing it"
    curl -fsSL https://astral.sh/uv/install.sh | bash
fi

cd $cur_dir
uv sync
uv run get-tpu.py "${args[@]}"
