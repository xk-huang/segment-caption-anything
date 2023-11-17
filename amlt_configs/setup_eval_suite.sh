#!/bin/bash
source ~/.bashrc

ORIGINAL_DIR="$(pwd)"
git clone --recursive https://github.com/xk-huang/vdtk.git /tmp/vdtk -b dev
cd /tmp/vdtk
git submodule update --init --recursive

apt-get update
sudo apt-get update
apt-get install git-lfs gawk
sudo apt-get install git-lfs gawk

git lfs install
git clone https://huggingface.co/xk-huang/vdtk-data
# git submodule init && git submodule update

rsync -avP ./vdtk-data/vdtk .
rm -rf vdtk-data

pip install --upgrade pip
pip install -e . POT==0.9.0  # POT=0.9.1 will take up all the memory with tf backend
pip install tensorflow==2.12.1  # Just fix one version of tf
pip install levenshtein==0.21.1
pip install openpyxl==3.1.2

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
cd "$ORIGINAL_DIR"
