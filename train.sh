#!/usr/bin/env bash
GPUS=$1
PORT=${PORT:-29600}

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    main.py ${@:2}
