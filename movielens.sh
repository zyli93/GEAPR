#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python3 src/main_zyli.py \
    --trial_id 1 \
    --epoch 20 \
    --batch_size 128 \
    --dataset "ml" \
    --save_n_epoch 5000 \
    --print_n_epoch 200 \
    --log_n_iter 100 \
    --learning_rate 0.001 \
    --l2_reg 0.01 \
    --embedding_size 128 \
    --regularization_weight 0.01
