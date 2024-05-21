#!/bin/sh

sudo apt update
sudo apt install -y python3.10-venv

git config --global credential.helper store

cd ~
python -m venv ~/Dev/venv/hf
source ~/Dev/venv/hf/bin/activate

pip install -U pip

# Install torch/TPU
pip install "torch~=2.3.0" "torch_xla[tpu]~=2.3.0" -f https://storage.googleapis.com/libtpu-releases/index.html

pip install transformers[sentencepiece] accelerate

# Add user to docker group
sudo usermod -aG docker $USER
