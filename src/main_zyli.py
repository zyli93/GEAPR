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

# from scipy.sparse import *
# from scipy.sparse 
import tensorflow as tf

from model import IRSModel
from train import trainer
from dataloader import DataLoader
from utils import check_flags, create_dirs


flags = tf.app.flags

# Run time
flags.DEFINE_bool("is_training", True, "The flag to run training or evaluation.")
flags.DEFINE_string('trial_id', '001', 'The ID of the current run.')
flags.DEFINE_integer('epoch', 300, 'Number of Epochs.')
flags.DEFINE_integer('batch_size', 64, 'Number of training instance per batch.')
flags.DEFINE_string('dataset', 'yelp', 'Input dataset name')
flags.DEFINE_string('yelp_city', 'lv', 'City data subset of yelp')
flags.DEFINE_integer('save_per_iter', 1000, 'Number of iterations per save.')
flags.DEFINE_integer('log_per_iter', 200, "Number of iterations per log.")
flags.DEFINE_bool('reshuffle', False, "Re-shuffle training data.")  # TODO: keep this?

# Hyperparam - Optimization
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('regularization_weight', 0.0001, 'Weight of L2 Regularizations.')
flags.DEFINE_integer('random_seed', 723, 'Random Seed.')

# Hyperparam - Model
flags.DEFINE_integer('embedding_dim', 64, 'Hidden embedding size.')
flags.DEFINE_integer('rep_dim', 32, 'Internal representation dimensions.')
flags.DEFINE_float("tao", 0.2, "Temperature constant!")
flags.DEFINE_integer("num_user_ctrd", 32, "Number of centroids for users.")
flags.DEFINE_integer("num_item_ctrd", 64, "Number of interest for items.")

# Model option
flags.DEFINE_string("corr_metric", "cos", "Correlation metrics for centroids.")  # "cos", "log"
flags.DEFINE_float("corr_weight", 0.1, "Correlation cost weight")

# MLP Encoder
flags.DEFINE_list('mlp_layers', None, "[comma sep. list] Structural context encoder layers.")
# flags.DEFINE_string('ae_item_enc_layers', "", "Item size AE structure.")
# flags.DEFINE_float("ae_weight", 0.1, "Auto encoder reconstruct error weight.")

# Graph Attention Network
# TODO

FLAGS = flags.FLAGS

def main():
    """entry of training or evaluation"""

    # check FLAGS correctness and directories
    check_flags(FLAGS)
    create_dirs(FLAGS)

    # data loader
    print("[IRS] loading dataset ...")
    dataloader = DataLoader(flags=FLAGS)

    # build graph
    print("[IRS] creating model and building graph ...")
    model = IRSModel(flags=FLAGS)

    # run trainer, evaluation included
    if FLAGS.is_training:
        print("[IRS] start running training ...")
        trainer(flags=FLAGS, model=model, dataloader=dataloader)
    else:
        print("[IRS] start running evaluation ...")
        raise NotImplementedError("evaluation not implemented")
    


if __name__ == '__main__':
    tf.app.run()
