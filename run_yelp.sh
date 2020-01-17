#!/usr/bin/env bash

#   Notes:
#       1 - batch_size: start from small
#       2 - save_model: False (for debugging and training)
#
#   Unset params
#       1 - dataset
#       2 - save_per_iter
#       3 - random_seed
#       4 - is_training
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

# CUDA_VISIBLE_DEVICES=$1 
python3 src/main_zyli.py \
    --ae_layers 64,32 \
    --trial_id 1 \
    --epoch 300 \
    --batch_size 128 \
    --yelp_city tor \
    --nosave_model \
    --log_per_iter 200 \
    --negative_sample_ratio 3 \
    --valid_set_size 256  --loss_type ranking \
    --learning_rate 0.0005 \
    --regularization_weight 0.0001 \
    --embedding_dim 64 \
    --hid_rep_dim 32 \
    --tao 0.5 \
    --num_total_item 9102 \
    --num_total_user 9582 \
    --ae_recon_loss_weight 0.01 \
    --gat_nheads 1 \
    --gat_ft_dropout 0.3 \
    --gat_coef_dropout 0.3 \
    --afm_dropout \
    --afm_num_total_user_attr 80 \
    --afm_num_field 8 \
    --num_user_ctrd 16 \
    --num_item_ctrd 16 \
    --corr_weight 0.01 \
    --candidate_k 10,5,20,30
