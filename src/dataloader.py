"""
    Data loader file

    @author: Zeyu Li <zyli@cs.ucla.edu>

    TODO:
        1 - only implemented yelp related datasets
        2 - implement get test data set!
"""

import pandas as pd
from scipy.sparse import load_npz

# from utils import *
from utils import load_pkl


# Global variables
YELP_PARSE = "./data/parse/yelp/"
YELP_CITY = YELP_PARSE + "citycluster/"
YELP_INTERACTION = YELP_PARSE + "interactions/"
YELP_TRAINTEST = YELP_PARSE + "train_test/"
YELP_GRAPH = "./data/graph/yelp/"

class DataLoader:
            #   2. rename business to item
    """docstring of DataLoader"""
    def __init__(self, flags):
        """DataLoader for loading two types of data: 
            - train/test
                - training positive instances
                - training negative sample candidates
                - testing instance positive sample
            - feature information
                - attribute information (user/item)
                - user-friendship graph (uf_graph)
                - user-structure-context graph (usc_graph)

        Args:
            flags - contains all the flags

        Notes:
            self.item_col_name:
                refers to "business" when dataset is "yelp"
        """
        self.f = flags
        self.nsr = self.f.negative_sample_ratio

        if self.f.dataset == "yelp":
            self.item_col_name = "business"
            # interaction_dir = YELP_INTERACTION + self.f.yelp_city + "/"
            train_test_dir = YELP_TRAINTEST + self.f.yelp_city + "/"
            city_dir = YELP_CITY + self.f.yelp_city + "/"
            graph_dir = YELP_GRAPH + self.f.yelp_city + "/"

            print("[Data loader] loading friendship and strc-ctx graphs",
                  "and user-friendship dict")
            self.uf_graph = load_npz(graph_dir + "uf_graph.npz")
            self.usc_graph = load_npz(graph_dir + "uf_sc_graph.npz")
            self.uf_dict = load_pkl(city_dir + "city_user_friend.pkl")

            print("[Data loader] loading train pos, train neg, and test instances.")
            # self.train_data = pd.read_csv(interaction_dir + "train.csv")
            # self.test_data = pd.read_csv(interaction_dir + "test.csv")
            # self.dev_data = pd.read_csv(interaction_dir + "dev.csv")
            self.train_pos = pd.read_csv(train_test_dir + "train_pos.csv")
            self.train_pos = self.train_pos.values

            self.train_neg = load_pkl(train_test_dir + "train_neg.pkl")
            self.test_instances = load_pkl(train_test_dir + "test_instances.pkl")

            # TODO:
            #   1. map all features to categorical
            #   2. convert all pandas dataframe to numpy array
            print("[Data loader] loading user/item attributes")
            self.user_attr = pd.read_csv(city_dir + "processed_city_user_profile.csv")
            self.item_attr = pd.read_csv(city_dir + "processed_city_business_profile.csv")

        else:
            self.item_col_name = None
            raise NotImplementedError("[DataLoader] Now only support yelp")

        self.set_to_dataset = {
            "train": self.train_pos,
            "test": self.test_instances,
        }


    def get_train_batch_iterator(self):
        """Create a train batch data iterator

        Yield:
            (iterator) of the dataset
            i - the index of the returned batch
            batch_users - (batch_size, ) batch of users
            batch_items_pos - (batch_size, ) batch of positive items
            batch_items_neg - (batch_size, self.nsr) batch of negative items
        """
        # define negagive sample function
        neg_sample_func = lambda x: np.random.choice(
            self.train_neg[x], size=self.nsr, replace=True)

        bs = self.f.batch_size
        total_batch = len(data) // self.f.batch_size
        for i in range(total_batch):
            batch = self.train_pos[i * bs: (i+1) * bs]
            batch_users = batch_pos[:, 0]
            batch_items_pos = batch_pos[:, 1]
            batch_items_neg = np.array([neg_sample_func[x] for x in batch_pos_users])
            yield (i, batch_users, batch_items_pos, batch_items_neg)


    def get_user_graphs(self, user_array):
        """get the graph information of users

        Args:
            user_array - numpy array of users to fetch data for
        Returns:
            uf_mat - user-friendship adjacency matrix
            usc_mat - user structural context info matrix
            uf_nbr - user-frienship neighborhood relationships

        TODO:
            1. if all three information is needed
        """
        uf_mat = self.uf_graph[user_array]
        usc_mat = self.usc_graph[user_array]
        uf_nbr = {k: self.uf_dict[k] for k in user_array}
        return uf_mat, usc_mat, uf_nbr


    def get_user_attributes(self, user_array):
        """get the user attributes matrixs

        Args:
            user_array - numpy array of users to featch data for
        Returns:
            user attribute submatrix
        """
        return self.user_attr.iloc[user_array]


    def get_item_attributes(self, item_array):
        """get the item attributes matrixs

        Args:
            item_array - numpy array of items to featch data for
        Returns:
            item attribute submatrix
        """
        return self.item_attr.iloc[item_array]


    def get_dataset_size(self):
        """get the size of datasets

        Return:
            size of train 
        """
        return len(self.train_pos)
