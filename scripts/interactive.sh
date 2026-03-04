#!/usr/bin/env bash

# Sleep for 7 hours
sleep 7h

# Run the Slurm interactive job
srun --partition=dgx-b200 --gpus=2 --ntasks=1 --cpus-per-task=32 --mem=768G --time=0-18:00:00 --sockets-per-node=1 --pty bash
