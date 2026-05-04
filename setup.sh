#!/bin/sh

sudo apt update
sudo apt install -y python3-virtualenv python-is-python3

git config --global credential.helper store

cd ~
rm -rf ~/Dev/venv/hf

# Add user to docker group
sudo usermod -aG docker $USER
