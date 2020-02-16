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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from geapr.model import IRSModel
from geapr.train import train
from geapr.dataloader import DataLoader
from utils import check_flags, create_dirs

# 2 = INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags

# Run time
flags.DEFINE_string('trial_id', '001', 'The ID of the current run.')
flags.DEFINE_integer('epoch', 300, 'Number of Epochs.')
flags.DEFINE_integer('batch_size', 64, 'Number of training instance per batch.')
flags.DEFINE_string('yelp_city', 'lv', 'City data subset of yelp')
flags.DEFINE_boolean("save_model", False, "Whether to save the model")
flags.DEFINE_integer('save_per_iter', 1000, 'Number of iterations per save.')
flags.DEFINE_integer('log_per_iter', 200, "Number of iterations per log.")
flags.DEFINE_integer('negative_sample_ratio', 3, "Negative sample ratio")
flags.DEFINE_string("loss_type", "ranking", "Choose from `binary` and `ranking`")
flags.DEFINE_boolean("separate_loss", False, "Whether to separate loss terms!")

# Hyperparam - Optimization
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('regularization_weight', 0.0001, 'Weight of L2 Regularizations.')
flags.DEFINE_integer('random_seed', 723, 'Random Seed.')

# Hyperparam - Model
flags.DEFINE_integer('embedding_dim', 64, 'Hidden embedding size.')
flags.DEFINE_integer('hid_rep_dim', 32, 'Internal representation dimensions.')
flags.DEFINE_integer("num_total_item", None, "Number of total items.")
flags.DEFINE_integer("num_total_user", None, "Number of total users.")


# Auto Encoder
flags.DEFINE_list('ae_layers', None, 
    "[comma sep. list] Structural context AE layers. No RAW dim, no MID dim.")

# Graph Attention Network
flags.DEFINE_integer('gat_nheads', 2, "Number of heads in GAT")
flags.DEFINE_float('gat_ft_dropout', 0.4, "Dropout rate of GAT feedforward net")
flags.DEFINE_float('gat_coef_dropout', 0.4, "Dropout rate of GAT coefficient mat")

# Attentional Factorization Machine
flags.DEFINE_boolean("afm_use_dropout", False, "Whether to use dropout in attentional FM")
flags.DEFINE_float("afm_dropout_rate", 0.3, "The dropout rate for attentional FM")
flags.DEFINE_integer("afm_num_total_user_attr", None, "Number of total user attributes")
flags.DEFINE_integer("afm_num_field", None, "Number of fields of user attributes")

# Geolocation features
flags.DEFINE_integer("num_lat_grid", 30, "Number of latitude grids.")
flags.DEFINE_integer("num_long_grid", 30, "Number of longitude grids.")

# Metrics
flags.DEFINE_list("candidate_k", None, "Evaluation Prec@k, Recall@k, MAP@k and NDCG@k")

FLAGS = flags.FLAGS


def main(args):
    """entry of training or evaluation"""

    print("== tf version: {} ==".format(tf.__version__))
    print(FLAGS.ae_layers)
    print(FLAGS.trial_id)
    print(FLAGS.loss_type)
    print(FLAGS.negative_sample_ratio)
    print(FLAGS.regularization_weight)
    print(FLAGS.embedding_dim)
    print(FLAGS.gat_nheads)
    print(FLAGS.afm_num_total_user_attr)

    # check FLAGS correctness and directories
    check_flags(FLAGS)
    create_dirs(FLAGS)

    print(FLAGS.ae_layers)

    # build graph
    """One thing that I learned is to build model before l
        loading data. Loading data won't be very troublesome.
        But building model will."""

    print("[IRS] creating model and building graph ...")
    model = IRSModel(flags=FLAGS)

    # data loader
    print("[IRS] loading dataset ...")
    dataloader = DataLoader(flags=FLAGS)

    # run training
    print("[IRS] start running training ...")
    train(flags=FLAGS, model=model, dataloader=dataloader)


if __name__ == '__main__':
    tf.compat.v1.app.run()
