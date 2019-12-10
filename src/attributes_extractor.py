"""User and Business attributes extractor for yelp

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

    Notes:
        1. All vacant default of features are set as CNT_DFL.

    TODO:
        1. load data from
        /local2/zyli/irs_fn/data/parse/yelp/preprocess/{user_profile.pkl, business_profile.pkl}
        2. for user:
            2.1 count the coverage of attributes that we are interested in
                - elite (e.g.: [2012, 2013])
                - review_count
                - funny/cool/useful
                - average_stars
                - yelp_since
            2.2 see if there are users that don't have above features
        3. for business:
            3.1 parse diff categories
            3.2 count cat coverage
            3.3 other attributes:
                - longitude ?
                - latitude ?
                - stars
                - review count
                - [isn't there a register time?]
        4. install user/business data features in table


"""

import sys

from datetime import datetime
import pandas as pd
from dateutil import parser

try:
    import _pickle as pickle
except ImportError:
    import pickle

# Global variables
INPUT_DIR = "data/parse/yelp/citycluster/"
OUTPUT_DIR = "data/parse/yelp/citycluster/"

CNT_DFL = 0
STAR_DFL = 3
DATE_DFL = "2019-12-01 00:00:01"
NAME_DFL = "No Name Business"
EMPTY_CATS = 'NoCategories'

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

    user_data = []

    # process users, NOTE: user new index starts with 1
    for _, prof_dict in user_profile.items():
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

        user_data.append(tmp_entry)

    # create dataframe
    df_user_profile = pd.DataFrame(user_data)

    # non-zero count attributes
    df_nonzero = df_user_profile.fillna(0).astype(bool).sum(axis=1)
    df_nonzero = df_nonzero / len(df_user_profile)
    print("\t[user] non-zero terms in `df_user_profile`")
    print(df_nonzero)

    print("\t[user] saving dataframe to {}".format(OUTPUT_DIR))
    df_user_profile.to_csv(OUTPUT_DIR + "{}/processed_city_user_profile.csv".format(city),
                           index=False)

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

    business_data = []

    bus_cat_dicts = {EMPTY_CATS: 0}

    # process users, NOTE: user new index starts with 1
    for _, prof_dict in business_profile.items():
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

        business_data.append(tmp_entry)

    # create dataframe
    df_business_profile = pd.DataFrame(business_data)

    # non-zero
    df_nonzero = df_business_profile.fillna(0).astype(bool).sum(axis=1)
    df_nonzero = df_nonzero / len(df_business_profile)
    print("\t[business] non-zero terms in `df_business_profile`")
    print(df_nonzero)

    print("\t[business] saving dataframe to {}".format(OUTPUT_DIR))
    df_business_profile.to_csv(OUTPUT_DIR + "{}/processed_business_user_profile.csv".format(city),
                               index=False)

    print("\t[business] saving categories dictionary")
    with open(OUTPUT_DIR + "{}/bus_cat_idx_dict.pkl".format(city), "rb") as fout:
        pickle.dump(bus_cat_dicts, fout)

    return df_nonzero


if __name__ == "__main__":
    if len(sys.argv) < 1 + 1:
        print("python {} [city]".format(sys.argv[0]))
        raise ValueError("Invalid input")

    city = sys.argv[1]
    assert city in ["lv", "phx", "tor", "all"], \
            "invalid city, should be `all`, `lv`, `phx`, or `tor`"
    city = ['lv', 'phx', 'tor'] if city == "all" else [city]

    for c in city:
        print("[attribute extractor] building user attributes {}".format(c))
        extract_user_attr(c)

        print("[attribute extractor] building business attributes {}".format(c))
        extract_business_attr(c)
