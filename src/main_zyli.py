#! /usr/bin/python3

"""Entry point of the magic model

    == Interpretable Recommender System with Friendship Network ==

    Three sides:
        User:
            1. Direct neighbors
            2. Structural neighbors
            3. Implicit network
        Item:
            1. Implicit network
        Centroids:
            1. User
            2. Item

    @Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    NOTES:
        1. train/test/dev ratios have been pre-set in data preparation
"""

import os
import tensorflow as tf

from model import IRSModel
from train import train
from dataloader import DataLoader
from utils import check_flags, create_dirs

# 2 = INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags

# Run time
flags.DEFINE_string('trial_id', '001', 'The ID of the current run.')
# flags.DEFINE_boolean("is_training", True, "The flag to run training or evaluation.")
flags.DEFINE_integer('epoch', 300, 'Number of Epochs.')
flags.DEFINE_integer('batch_size', 64, 'Number of training instance per batch.')
flags.DEFINE_string('dataset', 'yelp', 'Input dataset name')
flags.DEFINE_string('yelp_city', 'lv', 'City data subset of yelp')
flags.DEFINE_boolean("save_model", False, "Whether to save the model")
flags.DEFINE_integer('save_per_iter', 1000, 'Number of iterations per save.')
flags.DEFINE_integer('log_per_iter', 200, "Number of iterations per log.")
flags.DEFINE_integer('negative_sample_ratio', 3, "Negative sample ratio")
flags.DEFINE_integer('valid_set_size', 64, "Validation set size")
flags.DEFINE_string("loss_type", "ranking", "Choose from `binary` and `ranking`")

# Hyperparam - Optimization
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('regularization_weight', 0.0001, 'Weight of L2 Regularizations.')
flags.DEFINE_integer('random_seed', 723, 'Random Seed.')

# Hyperparam - Model
flags.DEFINE_integer('embedding_dim', 64, 'Hidden embedding size.')
flags.DEFINE_integer('hid_rep_dim', 32, 'Internal representation dimensions.')
flags.DEFINE_float("tao", 0.2, "Temperature constant!")
flags.DEFINE_integer("num_total_item", None, "Number of total items.")
flags.DEFINE_integer("num_total_user", None, "Number of total users.")


# Auto Encoder
flags.DEFINE_list('ae_layers', None, 
    "[comma sep. list] Structural context AE layers. No RAW dim, no MID dim.")
flags.DEFINE_float("ae_recon_loss_weight", 0.1, "AutoEncoder reconstruction loss")

# Graph Attention Network
flags.DEFINE_integer('gat_nheads', 2, "Number of heads in GAT")
flags.DEFINE_float('gat_ft_dropout', 0.4, "Dropout rate of GAT feedforward net")
flags.DEFINE_float('gat_coef_dropout', 0.4, "Dropout rate of GAT coefficient mat")

# Attentional Factorization Machine
flags.DEFINE_boolean("afm_dropout", False, "Whether to use dropout in attentional FM")
flags.DEFINE_integer("afm_num_total_user_attr", None, "Number of total user attributes")
flags.DEFINE_integer("afm_num_field", None, "Number of fields of user attributes")
# TODO: add this to run bash

# Centroid
flags.DEFINE_integer("num_user_ctrd", 32, "Number of centroids for users.")
flags.DEFINE_integer("num_item_ctrd", 64, "Number of interest for items.")
flags.DEFINE_float("corr_weight", 0.1, "Correlation cost weight")
flags.DEFINE_string("centroid_act", None, "Centroid activation function")

# Metrics
flags.DEFINE_list("candidate_k", None, "Evaluation Prec@k, Recall@k, MAP@k and NDCG@k")


FLAGS = flags.FLAGS


def main(args):
    """entry of training or evaluation"""

    print("== tf version: {} ==".format(tf.__version__))
    print(FLAGS.ae_layers)
    print(FLAGS.trial_id)
    print(FLAGS.loss_type)
    print(FLAGS.regularization_weight)
    print(FLAGS.embedding_dim)
    print(FLAGS.gat_nheads)
    print(FLAGS.afm_num_total_user_attr)

    # check FLAGS correctness and directories
    check_flags(FLAGS)
    create_dirs(FLAGS)

    print(FLAGS.ae_layers)
    # data loader
    print("[IRS] loading dataset ...")
    dataloader = DataLoader(flags=FLAGS)

    # build graph
    print("[IRS] creating model and building graph ...")
    model = IRSModel(flags=FLAGS)

    # run trainer, evaluation included
    if FLAGS.is_training:
        print("[IRS] start running training ...")
        train(flags=FLAGS, model=model, dataloader=dataloader)
    else:
        print("[IRS] start running evaluation ...")
        raise NotImplementedError("evaluation not implemented")


if __name__ == '__main__':
    tf.app.run()
