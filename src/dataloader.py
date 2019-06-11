"""
    Data loader file

    @author: Zeyu Li <zyli@cs.ucla.edu>
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import *
from sklearn.model_selection import train_test_split

try:
    import _pickle as pickle
except:
    import pickle


class DataLoader:
    def __init__(self, flags):
        """DataLoader for loading two types of data

        :param flags: contains all the flags
        """
        self.F = flags

        self.adj_dir = os.path.join(os.getcwd(), "graph", self.F.dataset)
        self.parse_dir = os.path.join(os.getcwd(), "parse", self.F.dataset)

        self.batch_index = 0

        self.trn, self.tstX, self.tstY = self._load_review_data()
        self.trn_size = self.trn.shape[0]
        self.tst_size = self.tstX.shape[0]

        # sdne loader
        if self.F.model_type == "sdne":
            self.u_adj, self.i_adj, self.u_nbr, self.i_nbr = \
                self._load_graph_data()

        # gat loader
        else:
            self.u_emb, self.i_emb, self.u_nbr, self.i_nbr, \
            self.u_nbr2, self.i_nbr2 = \
                self._load_graph_data()

    # ============================
    #       internal functions
    # ============================
    def _load_graph_data(self):
        """
        load dataset

        if model_type == sdne:
            return u_adj, i_adj, u_nbr, i_nbr
        else:
            return u_emb, i_emb, u_nbr, i_nbr, u_nbr2, i_nbr2
        """

        with open(self.adj_dir + "/u.nbr.pkl", "rb") as fin:
            u_nbr = pickle.load(fin)
        with open(self.adj_dir + "/i.nbr.pkl", "rb") as fin:
            i_nbr = pickle.load(fin)

        # sdne data loader
        if self.F.model_type == "sdne":
            u_adj = load_npz(self.adj_dir + "/u.adj.npz")
            i_adj = load_npz(self.adj_dir + "/i.adj.npz")
            return u_adj, i_adj, u_nbr, i_nbr

        # gat data loader
        else:
            # TODO: implement following two lines
            u_init_emb = None
            i_init_emb = None

            with open(self.adj_dir + "/u.nbr.2nd.pkl", "rb") as fin:
                u_nbr2 = pickle.load(fin)
            with open(self.adj_dir + "/i.nbr.2nd.pkl", "rb") as fin:
                i_nbr2 = pickle.load(fin)

            return u_init_emb, i_init_emb, u_nbr, i_nbr, \
                   u_nbr2, i_nbr2

    def _load_review_data(self):
        """Load train dataset, test dataset, test label
        """
        if self.F.dataset == "ml":
            df = pd.read_csv(self.parse_dir + "rt45.csv")
            data = df[["userId", "movieId"]].to_numpy()
            # TODO: split train test look at other papers
        elif self.F.dataset == " ":
            df = None
            # TODO: to implement
        else:
            df = None
            # TODO: to implement

        return trn, tst, tst_label

    def _get_trn_batch_sdne(self):
        """
        Create training data generator
        """
        total_batch = self.trn_size // self.F.batch_size + 1
        bs = self.F.batch_size
        for batch_index in range(0, total_batch):
            b_end = min((batch_index + 1) * bs, self.trn_size)
            batch_ui = self.trn[batch_index * bs: b_end]

            # batch_ui[0]: user id;
            # batch_ui[1]: item id.
            # TODO: see if separate nbr_size is needed
            batch_u_nbr = np.array(
                [np.random.choice(self.u_nbr[x], self.F.sdne_nbr_size)
                 for x in batch_ui[0]])
            batch_i_nbr = np.array(
                [np.random.choice(self.i_nbr[x], self.F.sdne_nbr_size)
                 for x in batch_ui[1]])

            batch_u_adj = self.u_adj[batch_ui[0]]
            batch_i_adj = self.i_adj[batch_ui[1]]

            yield batch_ui, batch_u_adj, batch_i_adj, batch_u_nbr, batch_i_nbr

    def _get_trn_batch_gat(self):
        # TODO
        raise NotImplementedError("To be implemented.")

    def _get_tst_batch_sdne(self):
        total_batch = self.tst_size // self.F.batch_size + 1
        bs = self.F.batch_size
        for batch_index in range(0, total_batch):
            b_end = min((batch_index + 1) * bs, self.trn_size)
            batch_ui = self.tstX[batch_index * bs: b_end]
            batch_label = self.tstY[batch_index * bs: b_end]

            # batch_ui[0]: user id;
            # batch_ui[1]: item id.
            # TODO: see if separate nbr_size is needed

            batch_u_adj = self.u_adj[batch_ui[0]]
            batch_i_adj = self.i_adj[batch_ui[1]]

            yield batch_ui, batch_u_adj, batch_i_adj, batch_label

    def _get_tst_batch_gat(self):
        # TODO
        raise NotImplementedError("To be implemented")

    # ============================
    #        external functions
    # ============================

    def batch_generator(self):
        if self.F.model_type == "sdne":
            return self._get_trn_batch_sdne()
        else:
            return self._get_trn_batch_gat()

    def shuffle(self):
        np.random.shuffle(self.trn)







