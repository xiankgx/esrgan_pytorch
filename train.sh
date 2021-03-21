#!/bin/bash

python train.py \
    --multi_gpu --mixed_precision \
    --batch_size 16 \
    --netG_checkpoint saved_models/netG-118000.pth \
    --netD_checkpoint saved_models/netD-118000.pth \
    --warmup_batches 0 \
    --n_epochs 5