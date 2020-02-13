"""
plot interpretability images
"""

import numpy as np

input_dir = "./temp/"

def process():
    # load data
    ft_file = input_dir + "feat.npy"
    afm1_file = input_dir + "afm1.npy"
    afm2_file = input_dir + "afm2.npy"
    gat_file = input_dir + "gat.npy"
    gt_file = input_dir + "gt.npy"
    user_file = input_dir + "user.npy"

    ft = np.load(ft_file)
    afm1 = np.load(afm1_file)
    afm2 = np.load(afm2_file)
    gat = np.load(gat_file)
    gt = np.load(gt_file)
    user_file = np.load(user_file)






if __name__ == "__main__":
    process()
