#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=output_file
#SBATCH --error=output_error
#SBATCH --mem-per-cpu=16G
#SBATCH -e output_error
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=1

module load eth_proxy

srun python train.py \
        --cuda \
        --data data/mixed/ \
        --dataset mixed \
        --n_layer 8 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 16 \
        --d_inner 128 \
        --dropatt 0.0 \
	--dropout 0.1 \
        --optim adam \
        --lr 1e-05 \
        --warmup_step 0 \
        --max_step 10000000 \
        --tgt_len 128 \
	--eval_tgt_len 128 \
	--ext_len 0 \
        --scheduler constant \
        --mem_len 0 \
	--attn_type 0 \
        --batch_size 64 \
        --eval-interval 100 \
	--mixing-rate 0.31 \
        --work_dir "$SCRATCH/LM-TFM" \
        --pre_lnorm \
        --accumulate-gradients 2 \
        --seed 1
