#!/bin/bash

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=40096
export WORLD_SIZE=$(nvidia-smi -L | wc -l)

# Launch the script
python train_FL.py --local_bs 32 --K 10