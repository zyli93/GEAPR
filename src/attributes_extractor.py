"""User and Business attributes extractor for yelp

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    Notes:
        1. All vacant default of features are set as CNT_DFL.
        2. load data from
        /local2/zyli/irs_fn/data/parse/yelp/preprocess/{user_profile.pkl, business_profile.pkl}
        3. for user:
            3.1 count the coverage of attributes that we are interested in
                - elite (e.g.: [2012, 2013])
                - review_count
                - funny/cool/useful
                - average_stars
                - yelp_since
            3.2 see if there are users that don't have above features
        4. for business:
            4.1 parse diff categories
            4.2 count cat coverage
            4.3 other attributes:
                - longitude
                - latitude
                - stars
                - review count
                - name
        5. install user/business data features in table

    TODO: 
        - user location by lat and lng
        - allow different number of attributes
"""

import sys

from datetime import datetime
import pandas as pd
from dateutil import parser

try:
    import _pickle as pickle
except ImportError:
    import pickle

from utils import dump_pkl

# Global variables
INPUT_DIR = "data/parse/yelp/citycluster/"
OUTPUT_DIR = "data/parse/yelp/citycluster/"

CNT_DFL = 0
STAR_DFL = 3
DATE_DFL = "2019-12-01 00:00:01"
NAME_DFL = "No Name Business"
EMPTY_CATS = 'NoCategories'

ATTR_CONFIG = "configs/user_attr_discrete.txt"

# Global mappings


def extract_user_attr(city):
    """extract user attributes

    Args:
        city - the city to profess

    Print-outs:
        df_nonzero - non zero number ratios of each attribute

    Return:
        df_nonzero - as above.
    """

    print("\t[user] loading user interaction from {}...".format(city))
    with open(INPUT_DIR + "{}/city_user_profile.pkl".format(city), "rb") as fin:
        user_profile = pickle.load(fin)

    user_data_csv = []
    user_data_pkl = {}

    # process users, NOTE: user new index starts with 1
    for uid, prof_dict in user_profile.items():
        # --- create feature area ---
        # after checking, each user has all attributes, review_count is non-zero
        tmp_entry = dict()
        u_elite = prof_dict.get('elite', [])
        tmp_entry['elite_count'] = len(u_elite)  # user elite
        tmp_entry['review_count'] = prof_dict.get('review_count', CNT_DFL)  # review_count
        tmp_entry['fans_count'] = prof_dict.get('fans', CNT_DFL)  # fans
        tmp_entry['funny_score'] = prof_dict.get('funny', CNT_DFL)  # funny
        tmp_entry['cool_score'] = prof_dict.get('cool', CNT_DFL)  # cool
        tmp_entry['useful_score'] = prof_dict.get('useful', CNT_DFL)  # useful
        tmp_entry['avg_stars'] = prof_dict.get('average_stars', STAR_DFL)  # average stars

        reg_yelp_date = prof_dict.get('yelping_since', DATE_DFL)  # yelping years
        delta_time = datetime.today() - parser.parse(reg_yelp_date)
        tmp_entry['yelping_years'] = delta_time.days // 365
        # --- end create feature area ---

        user_data_csv.append(tmp_entry)
        user_data_pkl[uid] = tmp_entry

    # create dataframe
    df_user_profile = pd.DataFrame(user_data_csv)

    # non-zero count attributes
    df_nonzero = df_user_profile.fillna(0).astype(bool).sum(axis=0)
    df_nonzero = df_nonzero / len(df_user_profile)
    print("\t[user] non-zero terms in `df_user_profile`")
    print(df_nonzero)

    print("\t[user] saving dataframe to {}".format(OUTPUT_DIR))
    df_user_profile.to_csv(
        OUTPUT_DIR+"{}/processed_city_user_profile.csv".format(city), index=False)
    dump_pkl(path=OUTPUT_DIR+"{}/processed_city_user_profile.pkl".format(city), 
        obj=user_data_pkl)

    return df_nonzero


def extract_business_attr(city):
    """extract business attributes

    Args:
        city - the city to profess

    Print-outs:
        df_nonzero - non zero number ratios of each attribute

    Store:
        bus_cat_dicts - business category

    Return:
        df_nonzero - as above.
    """

    print("\t[business] loading business interaction {}...".format(city))
    with open(INPUT_DIR + "{}/city_business_profile.pkl".format(city), "rb") as fin:
        business_profile = pickle.load(fin)

    business_data_csv = []
    business_data_pkl = {}

    bus_cat_dicts = {EMPTY_CATS: 0}

    # process users, NOTE: user new index starts with 1
    for bid, prof_dict in business_profile.items():
        # --- create feature area ---
        tmp_entry = dict()
        categories = prof_dict.get("categories")
        if not categories:  # one outlier without categories (recorded as `None`)
            tmp_entry['category_indices'] = [EMPTY_CATS]
        else:
            categories = categories.strip().split(", ")
            tmp_entry['category_indices'] = []
            for cat in categories:
                if cat not in bus_cat_dicts:
                    bus_cat_dicts[cat] = len(bus_cat_dicts)
                tmp_entry['category_indices'].append(bus_cat_dicts[cat])
        tmp_entry['review_count'] = prof_dict.get('review_count', CNT_DFL)  # review_count

        tmp_entry['stars'] = prof_dict.get('stars', STAR_DFL)
        tmp_entry['name'] = prof_dict.get('name', NAME_DFL)  # confirmed that all bus have names
        tmp_entry['longitude'] = prof_dict.get('longitude')
        tmp_entry['latitude'] = prof_dict.get('latitude')
        # --- end create feature area ---

        business_data_csv.append(tmp_entry)
        business_data_pkl[bid] = tmp_entry

    # create dataframe
    df_business_profile = pd.DataFrame(business_data_csv)

    # non-zero
    df_nonzero = df_business_profile.fillna(0).astype(bool).sum(axis=0)
    df_nonzero = df_nonzero / len(df_business_profile)
    print("\t[business] non-zero terms in `df_business_profile`")
    print(df_nonzero)

    print("\t[business] saving dataframe to {}".format(OUTPUT_DIR))
    df_business_profile.to_csv(
        OUTPUT_DIR+"{}/processed_city_business_profile.csv".format(city), index=False)
    dump_pkl(OUTPUT_DIR+"{}/processed_city_business_profile.pkl".format(city), business_data_pkl)

    print("\t[business] saving categories dictionary")
    dump_pkl(OUTPUT_DIR+"{}/bus_cat_idx_dict.pkl".format(city), bus_cat_dicts)

    return df_nonzero


def discretize_field_attr(city, num_bkt):
    """Discretize continuous fields to

    Starting from 1 instead of 0

    Args:

    Returns:
        c - city
        num_bkt - the number of buckets for embedding continuous values
            >0 for a `num_bkt` number of buckets
            -1 for a total discretize, i.e., take integers as discrete values

    avg_stars,cool_score,elite_count,fans_count,funny_score,review_count,
    useful_score,yelping_years
    """
    with open(ATTR_CONFIG, "r") as fin:
        lines = fin.readlines()
        discrete_attrs = set([x.strip() for x in lines])

    print("\t[user] discretize - loading user attrs")
    df = pd.read_csv(INPUT_DIR + "{}/processed_city_user_profile.csv".format(city))
    cols_disc_info = dict()
    distinct_ft_count = 1
    distinct_cols_list = []

    distinct_df_col_names = []
    distinct_df_cols = []

    for col in df.columns:
        # treat attribute as discrete variable
        if col in discrete_attrs:
            num_vals = df[col].unique()
            vals_map = dict(zip(num_vals, range(0, len(num_vals))))
            distinct_df_col_names.append(col+"_d_dist")
            distinct_df_cols.append(
                df[col].apply(lambda x: vals_map(x)+distinct_ft_count))
            # df.assign(col+"_dist", lambda x: vals_map(x)+distinct_ft_count)
            entry = {"bucket": False,
                    "value_map": vals_maps, "count": num_vals}
            distinct_ft_count += num_vals

        # treat attribute as continuous variable
        else:
            max_val, min_val = df[col].max(), df[col].min()
            gap = (max_val - min_val) / num_bkt
            distinct_df_col_names.append(col+"_c_dist")
            distinct_df_cols.append(
                df[col].apply(lambda x: int(((x-min_val) // gap + distinct_ft_count))))
            # df.assign(col + "_dist", lambda x: int((x // gap + distinct_ft_count)))
            entry = {"bucket": True,
                    "max_val": max_val, "min_val": min_val, "count": num_bkt,
                    "min_disc_token": distinct_ft_count,
                    "max_disc_token": distinct_ft_count + num_bkt}
            distinct_ft_count += num_bkt

        cols_disc_info[col]  = entry
        

    df_disc = pd.DataFrame(data=dict(zip(distinct_df_col_names, distinct_df_cols)))
    print("\t[user] discretize - saving dist. attr. and info to {}".format(OUTPUT_DIR))
    df_disc.to_csv(OUTPUT_DIR + "{}/processed_city_user_profile_dist.csv".format(city),
        index=False)
    dump_pkl(OUTPUT_DIR + "{}/cols_disc_info.pkl".format(city), cols_disc_info)


if __name__ == "__main__":
    if len(sys.argv) < 2 + 1:
        print("python {} [city] [bucket_num]".format(sys.argv[0]))
        raise ValueError("invalid input")

    city = sys.argv[1]
    num_bkt = int(sys.argv[2])

    assert city in ["lv", "phx", "tor", "all"], \
            "invalid city, should be `all`, `lv`, `phx`, or `tor`"
    assert num_bkt > 0 or num_bkt == -1, "invalid `bucket_num`, should be gt 0"

    city = ['lv', 'phx', 'tor'] if city == "all" else [city]

    for c in city:
        print("[attribute extractor] building user attributes {}".format(c))
        extract_user_attr(c)

        print("[attribute extractor] attribute to discrete values user")
        discretize_field_attr(c, num_bkt)

        # note: 
        #   did not implement business attribute extraction

        # print("[attribute extractor] building business attributes {}".format(c))
        # extract_business_attr(c)
        # print("[attribute extractor] attribute to discrete values business")
