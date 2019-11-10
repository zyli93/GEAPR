#! /usr/bin/python3

"""
    Utils functions

    @authors: Zeyu Li <zyli@cs.ucla.edu>
"""

import os
import datetime
import tensorflow as tf
try:
    import _pickle as pickle
except:
    import pickle

ACT_FUNC = {
    "relu": tf.nn.relu,
    "tanh": tf.nn.tanh,
    "lrelu": tf.nn.leaky_relu
}


def create_dir(dataset):
    # create logs file
    log_dir = os.getcwd() + "/logs/{}/".format(dataset)
    if os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # create checkpoints file
    ckpt_dir = os.getcwd() + "/checkpoints/{}/".format(dataset)
    if os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    # create performance file
    perf_dir = os.getcwd() + "/performance/{}/".format(dataset)
    if os.path.isdir(perf_dir):
        os.mkdir(perf_dir)

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def parse_ae_layers(struct_str):
    # parse layers size:
    #   E.G. "100,100,4" to [100, 100, 4]

    if struct_str == "":
        raise ValueError("Must fill in user/item AE layers!")

    return [int(x) for x in struct_str.strip().split(",")]


def check_flags(flags):
    if flags.emb_model not in ['sdne', 'gat']:
        raise ValueError("Invalid argument `emb_model`."
                         "Should be in `ae` or `gat`.")

    if flags.dataset not in ["ml", "so", "yp"]:
        raise ValueError("Invalid argument `dataset`."
                         "Should be in `ml`, `so`, or `yp`.")

    if flags.dis_metrics not in ["cos", "log"]:
        raise ValueError("Invalid argument `dis_metrics`."
                         "Should be in `cos`, `log`.")

    if flags.ctrd_act_func not in ACT_FUNC:
        raise ValueError("Invalid argument `ctrd_act_func`"
                         "Should be in `tanh`, `relu`, and `lrelu`.")

    # TODO: check correctness check


def get_activation_func(func_name):
    # Correctness check happened earlier
    return ACT_FUNC[func_name]


def build_msg(stage, **kwargs):
    time = datetime.now().isoformat()[:24]
    msg = ("[{},{}] ".format(stage, time))

    for key, value in kwargs.items():
        if isinstance(value, int):
            msg += " {}:{:d}".format(key, value)
        elif isinstance(value, float):
            msg += " {}:{:.6f}".format(key, value)
        else:
            TypeError("Error in value type {}".format(type(value)))

    return msg


def dump_pkl(path, obj):
    """helper to dump objects"""
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pkl(path):
    """helper to load objects"""
    with open(path, "rb") as fin:
        return pickle.load(fin)



