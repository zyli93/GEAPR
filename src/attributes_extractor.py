"""User and Business attributes extractor for yelp

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    Notes:
        1. All vacant default of features are set as CNT_DFL.
        2. load data from INPUT_DIR (see below)
        3. Only user self features and user loc features are used.
            Loc features are learned from business average.
        4. Business features are not used for this time.

    TODO:
        - user max/min lat/long not added yet
"""

import sys
from datetime import datetime
import pandas as pd
from dateutil import parser
import configparser
from sklearn.preprocessing import LabelEncoder

try:
    import _pickle as pickle
except ImportError:
    import pickle

from utils import dump_pkl, load_pkl

# Global variables
INPUT_DIR = "data/parse/yelp/citycluster/"
OUTPUT_DIR = "data/parse/yelp/citycluster/"
TRNTST_DIR = "data/parse/yelp/train_test/"

CNT_DFL = 0
STAR_DFL = 3
DATE_DFL = "2019-12-01 00:00:01"
NAME_DFL = "No Name Business"
EMPTY_CATS = 'NoCategories'

ATTR_CONFIG = "configs/user_attr_discrete.txt"
ATTR_CONFIG_V2 = "configs/columns_{}.ini"  # This is implemented by configparser


def load_configs(city):
    """Automatic load the configs of columns"""
    config = configparser.ConfigParser()
    config.read(ATTR_CONFIG_V2.format(city))
    return config


def extract_user_attr(city):
    """extract user attributes
    Args:

        city - the city to profess
    Save to disk:
        df_nonzero - non zero number ratios of each attribute
    Return:
        df_nonzero - as above.
    """

    print("\t[user] loading user interaction from {}...".format(city))

    user_profile = load_pkl(INPUT_DIR + "{}/city_user_profile.pkl".format(city))
    user_loc = load_pkl(INPUT_DIR + "{}/city_user_loc.pkl".format(city))

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
        tmp_entry['mean_lat'] = user_loc[uid]["mean_lat"]
        tmp_entry['mean_long'] = user_loc[uid]["mean_long"]

        reg_yelp_date = prof_dict.get('yelping_since', DATE_DFL)  # yelping years
        delta_time = datetime.today() - parser.parse(reg_yelp_date)
        tmp_entry['yelping_years'] = delta_time.days // 365
        # --- end create feature area ---

        user_data_csv.append(tmp_entry)
        user_data_pkl[uid] = tmp_entry

    # create data frame
    empty_head_entry = pd.DataFrame({'elite_count': 0, "review_count": CNT_DFL,
        "fans_count": CNT_DFL, "funny_score": CNT_DFL, "cool_score": CNT_DFL,
        "useful_score": CNT_DFL, "avg_stars": STAR_DFL, "yelping_years": 0,
        "mean_lat": CNT_DFL, "mean_long": CNT_DFL}, index=[0])
    df_user_profile = pd.DataFrame(user_data_csv)
    assert empty_head_entry.shape[1] == df_user_profile.shape[1]
    df_user_profile = pd.concat([empty_head_entry, df_user_profile],
                              axis=0, sort=True).reset_index(drop=True)
    print("\t[user] length of `df_user_profile` {}".format(df_user_profile.shape[0]))

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


def discretize_field_attr(city):
    """Discretize continuous fields to

    Starting from 1 instead of 0

    Args:

    Returns:
        c - city

        [DEPRECATED]
        num_bkt - the number of buckets for embedding continuous values
            >0 for a `num_bkt` number of buckets
            -1 for a total discretize, i.e., take integers as discrete values

    avg_stars,cool_score,elite_count,fans_count,funny_score,review_count,
    useful_score,yelping_years
    """

    col_configs = load_configs(city)

    print("\t[user] discretize - loading user attrs")
    df = pd.read_csv(INPUT_DIR + "{}/processed_city_user_profile.csv".format(city))
    cols_disc_info = dict()
    ft_idx_start = 1
    distinct_df_col_names, distinct_df_cols = [], []
    le = LabelEncoder()  # create for transforming CAT features

    for col in df.columns:
        # treat attribute as discrete variable
        # if col in discrete_attrs:
        if col in col_configs['CATEGORICAL']:
            distinct_df_cols.append(pd.Series(le.fit_transform(df[col]))+ft_idx_start)
            distinct_df_col_names.append(col+"_d_dist")
            num_vals = len(le.classes_)
            vals_map_to = le.transform(le.classes_) + ft_idx_start
            vals_map = dict(zip(le.classes_, vals_map_to))
            entry = {"bucket": False, "value_map": vals_map, "count": num_vals,
                     "max_idx": max(vals_map_to), "min_idx": min(vals_map_to)}
            ft_idx_start += num_vals

        # treat attribute as continuous variable
        # else:
        elif col in col_configs['NUMERICAL']:
            num_bkt = col_configs.getint("NUMERICAL", col)
            max_val, min_val = df[col].max(), df[col].min()
            # # Say goodbye to this stupid way of implementation
            # gap = (max_val - min_val) / num_bkt
            # distinct_df_col_names.append(col+"_c_dist")
            # distinct_df_cols.append(
            #     df[col].apply(lambda x: int(((x-min_val) // gap + distinct_ft_count))))
            # df.assign(col + "_dist", lambda x: int((x // gap + distinct_ft_count)))

            distinct_df_col_names.append(col + "_c_dist")
            distinct_df_cols.append(
                pd.cut(df[col], num_bkt,
                       labels=range(ft_idx_start, ft_idx_start+num_bkt)))
            entry = {"bucket": True,
                    "max_val": max_val, "min_val": min_val, "count": num_bkt,
                    "min_idx": ft_idx_start, "max_idx": ft_idx_start + num_bkt - 1}
            ft_idx_start += num_bkt
        else:
            raise KeyError("{} is NOT configured in `columns_{}.ini`".format(col, city))

        cols_disc_info[col] = entry

    df_disc = pd.DataFrame(data=dict(zip(distinct_df_col_names, distinct_df_cols)))
    print("\t[user] discretize - saving dist. attr. and info to {}".format(OUTPUT_DIR))
    df_disc.to_csv(OUTPUT_DIR + "{}/processed_city_user_profile_dist.csv".format(city),
        index=False)
    dump_pkl(OUTPUT_DIR + "{}/cols_disc_info.pkl".format(city), cols_disc_info)


def compute_user_avg_loc(city):
    """compute average latitude and longitude of businesses each user visited

    Arg:
        city - the city
    """
    print("\t[user] computing location features")
    # df = pd.read_csv(TRNTST_DIR + "{}/train_pos.csv".format(city))
    df = pd.read_csv(INPUT_DIR + "{}/user_business_interaction.csv".format(city))
    bus_profile = load_pkl(INPUT_DIR + "{}/city_business_profile.pkl".format(city))

    # df.assign(business_latitude=lambda x: bus_profile[x.business]["latitude"])
    # df.assign(business_longitude=lambda x: bus_profile[x.business]["longitude"])

    b_lat_dict = dict([(k, v["latitude"]) for k, v in bus_profile.items()])
    b_long_dict = dict([(k, v["longitude"]) for k, v in bus_profile.items()])

    df = df.assign(bus_lat=df.business.map(b_lat_dict))
    df = df.assign(bus_long=df.business.map(b_long_dict))

    # "ll": latitude and longitude
    print("\t[user] aggregating location (lat and long) by user")
    df_loc = df.groupby("user").agg({"bus_lat": ['max', 'min', 'mean'],
                                     "bus_long": ['max', 'min', 'mean']})

    # rename max, min, mean col to max_lat, min_lat, or mean_at. Same as `long`
    # while still maintaining the index as `user`
    user_lat = df_loc.bus_lat.reset_index()
    user_long = df_loc.bus_long.reset_index()
    user_loc = user_lat.join(user_long, on="user", how="outer",
                             lsuffix="_lat", rsuffix="_long")
    user_loc = user_loc.fillna(user_loc.mean())  # now `user` is column
    user_loc_dict = user_loc.set_index("user").to_dict(orient="index")
    dump_pkl(OUTPUT_DIR + "{}/city_user_loc.pkl".format(city), user_loc_dict)


if __name__ == "__main__":
    if len(sys.argv) < 1 + 1:
        print("python {} [city]".format(sys.argv[0]))
        raise ValueError("invalid input")

    city = sys.argv[1]
    assert city in ["lv", "phx", "tor", "all"]
    city = ['lv', 'phx', 'tor'] if city == "all" else [city]

    for c in city:
        print("[attribute extractor] computing user avg Lat and Long")
        compute_user_avg_loc(c)

        print("[attribute extractor] building user attributes {}".format(c))
        extract_user_attr(c)

        print("[attribute extractor] attribute to discrete values user")
        discretize_field_attr(c)
