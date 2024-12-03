#!/bin/sh

sudo apt update
sudo apt install -y python3-virtualenv python-is-python3

git config --global credential.helper store

cd ~
virtualenv ~/Dev/venv/hf
source ~/Dev/venv/hf/bin/activate

pip install -U pip

# Install torch/TPU
pip install "torch==2.5.1" "torchvision==2.5.1" --index-url https://download.pytorch.org/whl/cpu
pip install "torch_xla[tpu]~=2.5.1" -f https://storage.googleapis.com/libtpu-releases/index.html

pip install transformers[sentencepiece] accelerate

# Add user to docker group
sudo usermod -aG docker $USER
