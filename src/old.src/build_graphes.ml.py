"""
    Build user friendship graphs

    After this file, a few more parsed graph files would be generated
        - u.adj.npz, i.adj.npz (user/item adjacency matrix)
        - u.nbr.pkl, i.nbr.pkl (user/item neighbor pickle)
        - u.nbr.2nd.pkl, i.nbr.2nd.pkl (user/item

    @author: Zeyu Li <zyli@cs.ucla.edu>
"""

import os
import sys
from scipy.sparse import *
from sklearn.preprocessing import normalize
import numpy as np
import argparse

try:
    import _pickle as pickle
except:
    import pickle

from utils import load_pkl, dump_pkl


GRAPH_DIR = "/data/graph/"
PARSE_DIR = "/data/parse/"


def build_implicit_user_adj(dataset, threshold):
    # load data
    if dataset == "ml":
        u_mat = load_npz(PARSE_DIR + "ml/u.npz")
    elif dataset == "yelp/lasvegas":
        raise NotImplementedError()
    else:
        raise NotImplementedError("other dataset not implemented yet")

    # normalize over each user
    u_norm = normalize(u_mat, axis=1, norm="l1")
    
    # build similarity matrix
    u_sim = u_norm.dot(u_norm.T)

    # two ways: filter out or boolean
    # TODO: what are the two ways
    # u_sim[u_sim <= threshold] = 0  # empty out lower edges
    # u_adj = u_sim
    
    u_adj = (u_sim > threshold).astype(int)  # to 0 and 1
    
    return u_adj

# ===========================
#     Func for item graphs
# ===========================

# TODO: what is this? designed for MovieLens only?
def build_item_adj_ml(threshold, w):
    # ** whether to consider genre information **
    # process movie genre
    mg_mat = load_npz(PARSE_DIR + "ml/mg.npz")
    mg_norm = normalize(mg_mat, axis=1, norm="l1")
    mg_sim = mg_norm.dot(mg_norm.T)

    # process movie tags
    mt_mat = load_npz(PARSE_DIR + "ml/mt.npz")
    mt_norm = normalize(mt_mat, axis=1, norm="l1")
    mt_sim = mt_norm.dot(mt_norm.T)

    # build simiarity matrix
    m_sim = w * mg_sim + (1 - w) * mt_sim

    # two ways as well
    # m_sim[m_sim <= threshold] = 0  # empty out
    # ower edges
    # m_adj = m_sim

    m_adj = (m_sim > threshold).astype(int)  # to 0 and 1

    return m_adj


def build_augment_adj(adj_mat, rwr_order, rwr_rate):
    """Random Walk with Restart (RWR)

    Args:
        adj_mat - Adjacency matrix
        rwr_order - rwr orders
        rwr_rate - rwr with restart rate

    Return:
        result
    """

    # check if adj mat is square
    assert adj_mat.shape[0] == adj_mat.shape[1], "matrix not square"

    print("shape", adj_mat.shape)
    print("nnz", adj_mat.nnz)

    # if adj_mat.nnz / (adj_mat.shape[0] * adj_mat.shape[1]) > 0.1:
    #     adj_mat = (adj_mat.todense()).A

    # nam: normalized adjacency matrix
    print("augment matrix.\n\tnormalizing by rows")
    nam = normalize(adj_mat, axis=1, norm="l1")
    results = [nam]

    # 0 to rwr_order-1
    temp = nam
    for order in range(rwr_order):
        print("\tcomputing order {}".format(order + 1))
        temp = (1 - rwr_rate) * temp.dot(nam) + rwr_rate * nam
        results.append(temp)

    # sum up
    print("\tsumming up")
    aug_adj = np.sum(np.stack(results), axis=0)

    if issparse(aug_adj):
        nz_count = aug_adj.nnz
    else:
        nz_count = np.nonzero(aug_adj)[0].shape[0]

    ratio = nz_count / aug_adj.shape[0] ** 2
    print("\t after RWR, non-zero ratio: {}".format(ratio))

    return aug_adj


def build_neighbors(adj):
    """build neighbor by adjacency matrix

    return type:
        uid: array[neighbors]
    """
    n_user = adj.shape[0]
    if issparse(adj):
        neighbors = [adj.getrow(x).nonzero()[1]
                     for x in range(n_user)]
    else:
        neighbors = [adj[x].nonzero()[0] for x in range(n_user)]
    dict_neighbors = dict(zip(range(n_user + 1), neighbors))

    return dict_neighbors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset")
    parser.add_argument("-ut", "--u_threshold", type=float,
                        help="user similarity graph threshold")
    parser.add_argument("-it", "--i_threshold", type=float,
                        help="item similarity graph threshold")
    parser.add_argument("-rwr_o", "--rwr_order", type=int,
                        help="order of random walk with restart.")  # TODO: what is rwr?
    parser.add_argument("-rwr_c", "--rwr_constant", type=float,
                        help="constant of random walk with restart.")
    parser.add_argument("--ml_weight", type=float,
                        help="[To Fill In.]")  # TODO: what is this?
    args = parser.parse_args()


    # TODO: ? using argparse or manual parsing ??
    if len(sys.argv) < 2:
        print("python {} [dataset] [u_threshold]"
              "[rwr_order] [rwr_c] []".format(sys.argv[0]))
        sys.exit(1)
    
    # parse arguments
    dataset = sys.argv[1]
    u_threshold = float(sys.argv[2])
    i_threshold = float(sys.argv[3])

    rwr_order = int(sys.argv[4])
    rwr_c = float(sys.argv[5])
    ml_weight = float(sys.argv[6])  #

    adj_dir = GRAPH_DIR + dataset + "/"

    if not os.path.isdir(adj_dir):
        os.mkdir(adj_dir)

    print("building user adjacency matrix ...")
    u_adj = build_implicit_user_adj(args.dataset,
                                    args.u_threshold)

    if dataset == "ml":
        print("building item adjacency matrix ...")
        i_adj = build_item_adj_ml(i_threshold, ml_weight)
    else:
        raise NotImplementedError("Other functions not implemented.")

    # build augmented (RWR) adjacency matrix for u and i
    print("build RWR adjacency matrix ...")
    u_aug_adj = build_augment_adj(u_adj, rwr_order, rwr_c)
    i_aug_adj = build_augment_adj(i_adj, rwr_order, rwr_c)

    # save augmented (RWR) matrix
    print("saving user and item adjacency matrix")
    save_npz(adj_dir + "u.adj.npz", u_aug_adj)
    save_npz(adj_dir + "i.adj.npz", i_aug_adj)

    # create neighbors
    print("create neighbor linked list")
    u_nbr = build_neighbors(u_aug_adj)
    i_nbr = build_neighbors(i_aug_adj)

    # save neighbors to files
    print("dump user and item neighbors")
    with open(adj_dir + "u.nbr.pkl", "wb") as funbr, \
        open(adj_dir + "i.nbr.pkl", "wb") as finbr:
        pickle.dump(u_nbr, funbr)
        pickle.dump(i_nbr, finbr)

    # build second order aug adj mat on original aug adj mat
    print("build second order neighbors")
    u_aug_adj_2 = build_augment_adj(u_aug_adj, 1, 0)
    i_aug_adj_2 = build_augment_adj(i_aug_adj, 1, 0)

    # build second order neighbors
    print("create second order neighbors")
    u_nbr2 = build_neighbors(u_aug_adj_2)
    i_nbr2 = build_neighbors(i_aug_adj_2)

    # save neighbors to files
    print("dump user and item neighbors: 2nd order")
    with open(adj_dir + "u.nbr.2nd.pkl", "wb") as funbr, \
            open(adj_dir + "i.nbr.2nd.pkl", "wb") as finbr:
        pickle.dump(u_nbr2, funbr)
        pickle.dump(i_nbr2, finbr)

