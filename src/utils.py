"""Utils functions for irsfn

    @authors: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
from datetime import datetime
import tensorflow as tf

try:
    import _pickle as pickle
except ImportError:
    import pickle

ACT_FUNC = {"relu": tf.nn.relu, "tanh": tf.nn.tanh, "lrelu": tf.nn.leaky_relu}
OUTPUT_DIR = "./output/"


def create_dirs(f):
    """create directories

    create `output` directory if necessary
    create `ckpt` and `performance` directory if needed
    """
    make_dir(OUTPUT_DIR)
    make_dir_rec(OUTPUT_DIR + "/ckpt/")
    make_dir_rec(OUTPUT_DIR + "/performance/")


def make_dir(path):
    """make a director"""
    if not os.path.isdir(path):
        os.mkdir(path)


def make_dir_rec(path):
    """make a directory recursively, okay if exist"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def check_flags(f):
    """check validity of a few flag options and parse layers"""
    assert f.dataset in ["ml", "yelp"], "`dataset` should be `ml` or `yelp`"
    # assert f.corr_metric in ["cos", "log"], "`corr_metric` should be `cos` or `log`"
    # assert f.ctrd_act_func in ACT_FUNC, "`ctrd_act_func` should be `tanh`, `relu`, and `lrelu`"
    f.ae_layers = [int(x) for x in f.ae_layers]
    f.ae_layers.append(f.hid_rep_dim)
    assert f.ae_layers[-1] == f.hid_rep_dim, "ae_layers last should equal `hid_rep_dim`"

    f.candidate_k = [int(x) for x in f.candidate_k]
    return


def get_activation_func(func_name):
    # Correctness check happened earlier
    return ACT_FUNC[func_name]


def build_msg(stage, **kwargs):
    """build msg"""
    def build_single_msg(pref, **kw_dict):
        for key, value in kw_dict.items():
            if isinstance(value, int):
                pref += " {}:{:d}".format(key, value)
            elif isinstance(value, float):
                pref += " {}:{:.6f}".format(key, value)
            else:
                TypeError("Error in value type {}".format(type(value)))
        return pref

    assert stage in ["Trn", "Val", "Tst"], "Invalid `stage` in build_msg"
    time = datetime.now().isoformat()[:24]
    msg = ("[{},{}] ".format(stage, time))

    if stage == "Trn":
        return build_single_msg(msg, **kwargs)
    else:
        assert "eval_dict" in kwargs, "Has to put in eval_dict!"
        assert "epoch" in kwargs, "Has to put in epoch!"
        eval_dict = kwargs["eval_dict"]
        ep = kwargs["epoch"]
        msg_list = [build_single_msg(msg, ep=ep, k=k, **metrics) 
                    for k, metrics in eval_dict.items()]
        return "\n".join(msg_list)


def dump_pkl(path, obj):
    """helper to dump objects"""
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pkl(path):
    """helper to load objects"""
    with open(path, "rb") as fin:
        return pickle.load(fin)
