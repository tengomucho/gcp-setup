#!/bin/sh

sudo apt update
sudo apt install -y python3-virtualenv python-is-python3

git config --global credential.helper store

cd ~
virtualenv ~/Dev/venv/hf
source ~/Dev/venv/hf/bin/activate

pip install -U pip

# Install torch/TPU
pip install "torch==2.8.0" "torchvision==0.24.0" --index-url https://download.pytorch.org/whl/cpu
pip install "torch_xla[tpu]~=2.8.0" -f https://storage.googleapis.com/libtpu-releases/index.html

pip install transformers[sentencepiece] accelerate

# Add user to docker group
sudo usermod -aG docker $USER
