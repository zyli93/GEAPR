#! /usr/bin/python3

"""
    Entry of running the ml algorithm

    @Author: Zeyu Li <zyli@cs.ucla.edu>
"""

import os
import sys

import numpy as np
from scipy.sparse import *

import tensorflow as tf

from model import UnInteRec, DugrilpGAT
from train import trainer
from dataloader import DataLoader
from utils import create_dirs

flags = tf.app.flags

# Run time
flags.DEFINE_integer('epoch', 300, 'Number of Epochs.')
flags.DEFINE_integer('batch_size', 64, 'Number of training instance per batch.')
flags.DEFINE_string('dataset', 'yelp', 'Input dataset name')
flags.DEFINE_integer('save_n_iter', 100, 'Number of iterations per save.')
flags.DEFINE_integer('log_n_iter', 200, "Number of iterations per log.")
flags.DEFINE_string('reshuffle', False, "Re-shuffle training data.")  # TODO

# Hyperparam - Optimization
flags.DEFINE_float('learning_rate', 0.001, 'Learning Rate.')
flags.DEFINE_float('regularization_weight', 0.01, 'Weight of L2 Regularizations.')
flags.DEFINE_integer('random_seed', 723, 'Random Seed.')

# Hyperparam - Model
flags.DEFINE_integer('embedding_size', 64, 'Hidden Embedding Size.')
flags.DEFINE_string('trial_id', '001', 'The ID of the current run.')
flags.DEFINE_float("tao", 0.2, "Temperature constant!")
flags.DEFINE_integer("user_n_ctrd", 32, "Number of centroids for users.")
flags.DEFINE_integer("item_n_ctrd", 64, "Number of interest for items.")

# Model option
flags.DEFINE_string('emb_model', "ae", "Node embedding model.")  # "ae" or "gat"
flags.DEFINE_string("corr_metric", "cos", "Correlation metrics.")  # "cos", "log"
flags.DEFINE_float("corr_weight", 0.1, "Correlation cost weight")
flags.DEFINE_string("ctrd_act_func", "relu", "Activation function for centroid.")  # "relu", "tanh", "lrelu"

# Auto Encoder
# flags.DEFINE_string('ae_user_enc_layers', "", "User side AE structure.")
# flags.DEFINE_string('ae_item_enc_layers', "", "Item size AE structure.")
# flags.DEFINE_float('ae_beta_value', 2, "", "Beta value for SDNE AutoEncoder.") # TODO: figure out what's a good beta
# flags.DEFINE_float('ae_reg_weight', 0.01, 'The weight of L2-regularization in AE.')
# flags.DEFINE_float("ae_weight", 0.1, "Auto encoder reconstruct error weight.")
# flags.DEFINE_integer("sdne_nbr_size", 5, "Neighbor size of SDNE.")  # TODO: find a good neighbor size

# Graph Attention Network
# TODO

FLAGS = flags.FLAGS


def main(argv):

    # check FLAGS correctness
    check_flags(FLAGS)

    # create directories
    create_dirs(FLAGS.dataset)

    # data loader
    print("loading dataset ...")

    # create model
    if FLAGS.emb_model == "ae":
        # parse Auto Encoder layers, [0-user, 1-item]
        ae_layers = [parse_ae_layers(x) + [FLAGS.embedding_size]
                     for x in [FLAGS.ae_user_enc_layers,
                               FLAGS.ae_item_enc_layers]]

        dl = DataLoader(flags=FLAGS)

        model = UnInteRec(flags=FLAGS, ae_layers=ae_layers)

        # run trainer, evaluation included
        trainer(flags=FLAGS, model=model, data_loader=dl)

    elif FLAGS.emb_model == "sdne":
        raise NotImplementedError("DugrilpSDNE NOT implemented yet.")

    elif FLAGS.emb_model == "gat":
        raise NotImplementedError("DugrilpGAT NOT implemented yet.")

    else:
        raise ValueError("Invalid argument for `emb_model`!")


if __name__ == '__main__':
    tf.app.run()

