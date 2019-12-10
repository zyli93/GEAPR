"""
    Data loader file

    @author: Zeyu Li <zyli@cs.ucla.edu>

    TODO:
        only implemented yelp related datasets
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# from utils import *
from utils import load_pkl


# Global variables
YELP_PARSE = "./data/parse/yelp/"
YELP_CITY = YELP_PARSE + "citycluster/"
YELP_INTERACTION = YELP_PARSE + "interactions/"
YELP_GRAPH = "./data/graph/yelp/"

class DataLoader:
    """docstring of DataLoader"""
    def __init__(self, flags):
        """DataLoader for loading two types of data

        1. load user-friendship graph (uf_graph)
        2. load user-structure-context graph (usc_graph)
        3. load user-item interactions (dataset)
        4. load user features
        5. load item features

        :param flags: contains all the flags

        """
        self.f = flags

        if self.f.dataset == "yelp":
            interaction_dir = YELP_INTERACTION + self.f.yelp_city + "/"
            city_dir = YELP_CITY + self.f.yelp_city + "/"
            graph_dir = YELP_GRAPH + self.f.yelp_city + "/"

            print("[Data loader] loading friendship and strc-ctx graphs and user-friendship dict")
            self.uf_graph = load_npz(graph_dir + "uf_graph.npz")
            self.usc_graph = load_npz(graph_dir + "uf_sc_graph.npz")
            self.uf_dict = load_pkl(city_dir + "city_user_friend.pkl")

            print("[Data loader] loading train, test, and dev data")
            self.train_data = pd.read_csv(interaction_dir + "train.csv")
            self.test_data = pd.read_csv(interaction_dir + "test.csv")
            self.dev_data = pd.read_csv(interaction_dir + "dev.csv")

            print("[Data loader] loading user/item attributes")
            self.user_attr = load_pkl(city_dir + "processed_city_business_profile.pkl")
            self.item_attr = load_pkl(city_dir + "processed_city_business_profile.pkl")

        else:
            raise NotImplementedError("now only support yelp")

        self.batch_index = 0

        self.set_to_dataset = {
            "train": self.train_data,
            "test": self.test_data,
            "dev": self.dev_data
        }


    # ============================
    #        external functions
    # ============================

    def data_batch_generator(self, set_):
        """Create a train batch generator
        Args:
            set_ - `train`, `test`, or `dev`, the set to create iterator for.

        Yield:
            (iterator) of the dataset
        """
        data = self.set_to_dataset[set_]
        bs = self.f.batch_size
        total_batch = len(data) // self.f.batch_size
        for i in range(total_batch):
            subdf = data.iloc[i * bs: (i+1) * bs]
            label = subdf['label'].values
            user = subdf['user'].values
            business = subdf['business'].values
            yield (user, business, label)


    def get_user_graph(self, user_array):
        """get the graph information of users

        Args:
            user_array - numpy array of users to fetch data for

        Return:
            uf_mat - user-friendship adjacency matrix
            usc_mat - user structural context info matrix
            uf_nbr - user-frienship neighborhood relationships
        """
        uf_mat = self.uf_graph[user_array]
        usc_mat = self.usc_graph[user_array]
        uf_nbr = {k: self.uf_dict[k] for k in user_array}
        return uf_subgrf, usc_subgrf, uf_subdict


    def get_user_attributes(self, user_array):
        """get the user attributes matrixs

        Args:
            user_array - numpy array of users to featch data for

        Return:
        """
        # TODO

    def get_business_attributes(self, business_array):
        """get the user attributes matrixs

        Args:
            user_array - numpy array of users to featch data for

        Return:
        """
        # TODO



