#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
/home/hossein/miniconda3/envs/main/bin/python /home/hossein/github/DREAMS/main.py \
--lr 0.0000005 \
--batch_size 16 \
--epochs 50 \
