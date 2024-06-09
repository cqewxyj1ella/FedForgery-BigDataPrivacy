#!/bin/bash

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=40096
export WORLD_SIZE=$(nvidia-smi -L | wc -l)

# Launch the script
python -m torch.distributed.launch --nproc_per_node=$WORLD_SIZE train_distributed.py --global_bs 64