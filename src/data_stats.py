"""Helper for data collection

Load the data and compute the number of different features in the dataset
"""

import sys
import pandas as pd
from utils import load_pkl


def user_business_count(tt_dir):
    """User and Business cardinality count"""
    trn_pos = pd.read_csv(tt_dir+"train_pos.csv")
    trn_neg = load_pkl(tt_dir+"test_instances.pkl")

    u_trn_max, u_trn_min = int(trn_pos.user.max()), int(trn_pos.user.min())
    b_trn_max, b_trn_min = int(trn_pos.business.max()), int(trn_pos.business.min())
    print("user train max - {}, min - {}".format(u_trn_max, u_trn_min))
    print("business train max - {}, min - {}".format(b_trn_max, b_trn_min))

    u_tst_set = list(trn_neg.keys())
    if min(u_tst_set) >= u_trn_min:
        print("user test min included")
    else:
        print("user test min {}".format(min(u_tst_set)))

    if max(u_tst_set) <= u_trn_max:
        print("user test max included")
    else:
        print("user test max {}".format(max(u_tst_set)))

    b_tst_set = []
    for k, v in trn_neg.items():
        b_tst_set += list(v)
    b_tst_set = set(b_tst_set)

    if min(b_tst_set) >= b_trn_min:
        print("business test min included")
    else:
        print("business test min {}".format(min(b_tst_set)))

    if max(b_tst_set) <= b_trn_max:
        print("business test max included")
    else:
        print("business test max {}".format(max(b_tst_set)))


def user_attr_count(cc_dir):
    """User attribute count"""
    col_dict = load_pkl(cc_dir+"cols_disc_info.pkl")
    print("#. of fields: {}".format(len(col_dict.keys())))
    for k, v in col_dict.items():
        print("\tfeature {} - count {}, bkt {}, max {}, min {}"
              .format(k, v['count'], v['bucket'], v['max_idx'], v['min_idx']))
    total_feature = sum([x['count'] for x in col_dict.values()])
    print(total_feature)


def main(cc_dir, tt_dir):
    print("user business count")
    user_business_count(tt_dir)

    print("user attribute count")
    user_attr_count(cc_dir)


if __name__ == "__main__":
    if len(sys.argv) < 1 + 1:
        print("Usage:")
        print("\tpython {} [yelp_city]".format(sys.argv[0]))
        sys.exit()

    all_cities = ["lv", "tor", "phx"]
    cities = sys.argv[1]
    assert cities in all_cities or cities == "all"
    cities = all_cities if cities == "all" else [cities]
    for city in cities:
        cc_dir = "./data/parse/yelp/citycluster/{}/".format(city)  # cc: city cluster
        tt_dir = "./data/parse/yelp/train_test/{}/".format(city)
        main(cc_dir, tt_dir)