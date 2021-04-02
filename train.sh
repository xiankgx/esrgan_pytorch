#!/bin/bash

python train.py \
    --multi_gpu \
    --mixed_precision \
    --batch_size 32 \
    --warmup_batches 500 \
    --hr_height 208 \
    --hr_width 176 \
    --residual_blocks 23 \
    --num_upsample 4 \
    --n_epochs 100
