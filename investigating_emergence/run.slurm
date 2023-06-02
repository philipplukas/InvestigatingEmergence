#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=output_file
#SBATCH --mem-per-cpu=16G
#SBATCH -e output_error

srun python train.py \
        --cuda \
        --data data/ctl/ \
        --dataset ctl \
        --n_layer 6 \
        --d_model 128 \
        --n_head 8 \
        --d_head 16 \
        --d_inner 256 \
        --dropatt 0.0 \
	--dropout 0.1 \
        --optim adam \
        --lr 1e-04 \
        --warmup_step 0 \
        --max_step 10000000 \
        --tgt_len 128 \
	--eval_tgt_len 128 \
	--ext_len 0 \
        --scheduler cosine \
        --mem_len 0 \
	--attn_type 0 \
        --batch_size 128 \
        --eval-interval 100
