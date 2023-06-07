#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=output_file
#SBATCH --error=output_error
#SBATCH --mem-per-cpu=16G
#SBATCH -e output_error

srun python train.py \
        --cuda \
        --data data/mixed/ \
        --dataset ctl \
        --n_layer 12 \
        --d_model 512 \
        --n_head 8 \
        --d_head 16 \
        --d_inner 256 \
        --dropatt 0.0 \
	--dropout 0.1 \
        --optim adam \
        --lr 1e-04 \
        --warmup_step 0 \
        --max_step 10000000 \
        --tgt_len 512 \
	--eval_tgt_len 128 \
	--ext_len 0 \
        --scheduler constant \
        --mem_len 0 \
	--attn_type 0 \
        --batch_size 128 \
        --eval-interval 100
