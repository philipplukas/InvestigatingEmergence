#!/bin/bash

module load eth_proxy

# -r is used to limite maximal amount of tasks working in parallel 
ct -s -m localhost wandb sweep sweeps/sweep.yml -r 6 -t 72:00:00
