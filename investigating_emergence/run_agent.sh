#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=output_file
#SBATCH --error=output_error
#SBATCH --mem-per-cpu=16G
#SBATCH -e output_error
#SBATCH --time=72:00:00

module load eth_proxy

wandb agent investigating-emergence/investigating-emergence/ujfqzcgo
exit 0
