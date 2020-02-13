#!/usr/bin/env bash

#   Notes:
#       1 - batch_size: start from small
#       2 - save_model: False (for debugging and training)
#       3 - is_training: always true
#
#   Unset params
#       1 - dataset
#       2 - save_per_iter
#       3 - random_seed
#       4 - is_training
#       5 - ctrd_activation
#
#   Key hyper-params
#       1 - negative_sample_ratio - [3] (4, 5, ...)
#       2 - loss_type - "ranking" v.s. "binary"
#       3 - ae_layers
#       4 - gat_nheads - [1]
#   
#   Tuning hyper-params
#       1 - learning_rate
#       2 - embedding_dim & hid_rep_dim
#       3 - tao [0.5]
# 

CUDA_VISIBLE_DEVICES=1 python3 src/main_zyli.py \
    --noseparate_loss \
    --ae_layers 64,48 \
    --is_training \
    --trial_id 2001 \
    --epoch 7 \
    --batch_size 256 \
    --yelp_city tor \
    --nosave_model \
    --log_per_iter 200 \
    --negative_sample_ratio 105 \
    --valid_set_size 256  --loss_type binary\
    --learning_rate 0.001 \
    --regularization_weight 0.1 \
    --embedding_dim 64 \
    --hid_rep_dim 32 \
    --tao 0.7 \
    --num_total_item 9102 \
    --num_total_user 9582 \
    --ae_recon_loss_weight 0.01 \
    --gat_nheads 1 \
    --gat_ft_dropout 0.3 \
    --gat_coef_dropout 0.3 \
    --afm_use_dropout \
    --afm_dropout_rate 0.3 \
    --afm_num_total_user_attr 140 \
    --afm_num_field 10 \
    --num_user_ctrd 64 \
    --num_item_ctrd 64 \
    --ctrd_corr_weight 0.001 \
    --candidate_k 10,20,30,40,50,60,70,80,90,100 \
    --num_lat_grid 30 \
    --num_long_grid 30 \
    --task "inter"
