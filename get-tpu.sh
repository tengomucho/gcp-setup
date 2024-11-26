#!/bin/bash

args=("$@")
cur_dir=$(dirname "$0")

if [ ! -d "$cur_dir/.venv-get-tpu" ]; then
    echo "Error: Directory $cur_dir/.venv-get-tpu does not exist, creating it"
    python3 -m venv $cur_dir/.venv-get-tpu
    source $cur_dir/.venv-get-tpu/bin/activate

    pip install -U pip
    pip install typer
fi


source $cur_dir/.venv-get-tpu/bin/activate

python $cur_dir/get-tpu.py "${args[@]}"
