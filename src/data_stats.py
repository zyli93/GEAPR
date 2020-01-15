import pandas as pd
import pickle
from utils import load_pkl

city = "lv"
cc_dir = "./data/parse/yelp/citycluster/{}/".format(city) # cc: city cluster
tt_dir = "./data/parse/yelp/train_test/{}/".format(city)

def user_count():
    trn_pos = pd.read_csv(tt_dir+"train_pos.csv")
    print("user max train", trn_pos.user.max())
    trn_neg = load_pkl(tt_dir+"test_instances.pkl")
    print("user max test", max(trn_neg.keys()))


