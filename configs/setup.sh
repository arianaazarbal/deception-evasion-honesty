# Tested on the pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
# docker image, but should work on any recent platform
# with CUDA support
# source activate  # Commented out - only needed if using conda
apt-get update
apt-get install -y rsync
pip install uv

# Safest to install flash attention before other packages
uv pip install --system torch
uv pip install --system -e '.[dev]'
