"""Build user friendship neighbor graphs and structural context 

    @author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    This file generates a few more parsed graph archives
        - user structural context graph (usc_graph)
        - user friendship graph (uf_graph)
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

GRAPH_DIR = "./data/graph/"
PARSE_DIR = "./data/parse/"


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


def load_user_friend(dir_):
    """load user friendship graphes [new]

    Args:
        dir_ - directory to the processed files

    Return:
        uf_dict - user friendship matrix created for neighbors
        uf_graph - [csr_sparse matrix] adjacency matrix of uf_graph
    """
    load_pkl(dir_ + "city_user_friend.pkl")

    n_users = len(uf_dict)
    print("\t[Building graph] number of users: {}".format(n_users))
    row, col = [], []
    for u1, u2_list in uf_dict.items():
        row += [u1] * len(u2_list)
        col += u2_list
    data = [1] * len(row)
    uf_graph = csr_matrix((data, (row, col)), shape=(n_users, n_users))

    return uf_dict, uf_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="The dataset to work on. For yelp, use `yelp-lv`")
    parser.add_argument("--yelp_city", help="ABBRVs. of cities of yelp. Only useful when --dataset=yelp")
    # parser.add_argument("--u_threshold", type=float, help="user similarity threshold to build graph")
    # parser.add_argument("--i_threshold", type=float, help="item similarity threshold to build graph")
    parser.add_argument("--rwr_order", type=int, help="order of random walk with restart.")
    parser.add_argument("--rwr_constant", type=float, help="constant of random walk with restart.")
    parser.add_argument("--use_sparse_mat", type=bool, help="whether to use sparse matrix")
    args = parser.parse_args()

    # TODO: what is `i_threshold`

    # check parameters
    assert args.rwr_constant > 0 and rwr_constant < 1, "invalid rwr_constant value"
    assert args.dataset in ["yelp", "movielens"], "invalid `dataset` value, should be yelp or movielens"
    assert args.yelp_city in ["lv", "tor", "phx"], "invalid `yelp_city` value, should be `lv`, `tor`, or `phx`"

    if args.dataset == "yelp":
        input_dir = PARSE_DIR + "yelp/citycluster/" + args.yelp_city + "/"
        output_dir = GRAPH_DIR + "yelp/" + args.yelp_city + "/"
        # example output: ./data/graph/yelp/lv/..."
        print("[Building graphes] processing {}-{} ...".format(args.dataset, args.yelp_city))
    else:
        input_dir = PARSE_DIR + "ml/"
        output_dir = GRAPH_DIR + "ml/" 
        print("[Building graphes] processing {} ...".format(args.dataset))

    print("[Building graphes] output to {}".format(output_dir))
    make_dir_rec(output)

    # NOTES: not building the item side thing for now

    # load user friend matrix
    print("\t[Building graphes] loading user friendship")
    uf_dict, uf_graph = load_user_friend(input_dir)

    # build augmented (RWR) adjacency matrix for u and i
    print("\t[Building graphes] build RWR adjacency matrix ...")
    uf_sc_graph = build_augment_adj(uf_graph, args.rwr_order, args.rwr_constant)

    # save augmented (RWR) matrix
    print("\t[Building graphes] saving user friend matrix and user structural context matrix")
    if args.use_sparse:
        save_npz(output_dir + "uf_graph.npz", uf_graph)
        save_npz(output_dir + "uf_sc_graph.npz", uf_sc_graph)
    else:
        np.savez(output_dir + "uf_graph.npz", uf_graph)
        np.savez(output_dir + "uf_sc_graph.npz", uf_sc_graph)
    print("[Building graphes] finished creating graphes") 
