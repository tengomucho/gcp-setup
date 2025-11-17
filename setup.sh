#!/bin/sh

sudo apt update
sudo apt install -y python3-virtualenv python-is-python3

git config --global credential.helper store

cd ~
rm -rf ~/Dev/venv/hf
virtualenv ~/Dev/venv/hf
source ~/Dev/venv/hf/bin/activate

pip install -U pip

# Install torch/TPU
pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cpu
pip install scipy
pip install --pre torch_xla[pallas] --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install transformers[sentencepiece] accelerate

# Add user to docker group
sudo usermod -aG docker $USER
