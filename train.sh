#!/bin/bash

python train.py \
    --multi_gpu --mixed_precision \
    --batch_size 16