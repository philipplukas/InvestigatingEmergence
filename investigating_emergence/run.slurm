#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=output_file
#SBATCH --error=output_error
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00

srun python train.py \
        --cuda \
        --data data/mixed/ \
        --dataset ctl \
        --n_layer 12 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2028 \
        --dropatt 0.0 \
	--dropout 0.1 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 10000000 \
        --tgt_len 512 \
	--eval_tgt_len 128 \
	--ext_len 0 \
        --scheduler inv_sqrt \
        --mem_len 0 \
	--attn_type 2 \
        --batch_size 22 \
        --eval-interval 100 \
        --mixing-rate 1 \
	--log-interval 100
