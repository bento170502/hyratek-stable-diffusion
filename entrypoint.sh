#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate ldm

# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Execute the command passed to docker run
exec "$@"