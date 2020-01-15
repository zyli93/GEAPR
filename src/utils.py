#!/usr/bin/python3

"""
    Utils functions for irsfn

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

OUTPUT_DIR = "./output/"

def create_dirs(f):
    """create directories

    create `output` directory if necessary
    create `ckpt` and `performance` directory if needed
    """
    make_dir(OUTPUT_DIR)
    make_dir_rec(OUTPUT_DIR + "/{}/ckpt/".format(f.trial_id))
    make_dir_rec(OUTPUT_DIR + "/{}/performance/".format(f.trial_id))


def make_dir(path):
    """make a director"""
    if not os.path.isdir(path):
        os.mkdir(path)

def make_dir_rec(path):
    """make a directory recursively, okay if exist"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def parse_ae_layers(struct_str):
    """[Not used]"""
    # parse layers size:
    #   E.G. "100-100-4" to [100, 100, 4]
    assert len(struct_str) > 0, "Invalid MLP layers"
    return [int(x) for x in struct_str.strip().split("-")]


def check_flags(f):
    """check validity of a few flag options and parse layers"""
    assert f.dataset in ["ml", "yelp"], "`dataset` should be `ml` or `yelp`"
    assert f.corr_metric in ["cos", "log"], "`corr_metric` should be `cos` or `log`"
    assert f.ctrd_act_func in ACT_FUNC, "`ctrd_act_func` should be `tanh`, `relu`, and `lrelu`"
    f.ae_layers = [int(x) for x in f.ae_layers]
    f.ae_layers.append(f.rep_dim)
    assert f.ae_layers[-1] == f.hid_rep_dim, "ae_layers last should equal `hid_rep_dim`"

    f.candidate_k = [int(x) for x in f.candidate_k]
    return


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



