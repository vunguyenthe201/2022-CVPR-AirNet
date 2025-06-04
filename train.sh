#!/bin/bash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all 8 GPUs

# Directory paths - Update these to match your data
DATA_ROOT="../datasets/Universal_dataset"  # Relative path to Universal dataset
SAVE_DIR="./ckpt"

mkdir -p $SAVE_DIR

python train.py \
    --world-size 8 \
    --universal-dataset \
    --data-root $DATA_ROOT \
    --save-dir $SAVE_DIR \
    --batch-size 2 \
    --base-batch-size 4 \
    --gradient-accumulation-steps 2 \
    --epochs 200 \
    --lr 1e-4 \
    --use-amp \
    --use-checkpointing \
    --use-fft \
    --weighted-sampling \
    --lambda-cycle 10.0 \
    --lambda-identity 5.0 \
    --lambda-contrast 1.0 \
    --perceptual-weight 0.5 \
    --frequency-weight 1.0 \
    --tv-weight 0.1 \
    --optimizer adamw \
    --lr-scheduler cosine \
    --warmup-epochs 5 \
    --use-ema \
    --validate-with-ema \
    --augment \
    --num-workers 4 \
    --save-freq 10 \
    --val-freq 5 \
    --visualize-freq 100 \
    --use-tensorboard \
    --plot-metrics \
    --clip-grad \
    --clip-value 1.0 \
    --set-grads-to-none \
    --persistent-workers \
    --aggressive-memory-cleanup \
    --task restoration \
    --tiled-validation

echo "Training launched successfully!" 