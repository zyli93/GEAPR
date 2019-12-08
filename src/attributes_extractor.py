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

import os
import sys

import pandas as pd
from datetime import datetime
from dateutil import parser

try:
    import _pickle as pickle
except:
    import pickle

# Global variables
INPUT_DIR = "../data/parse/yelp/citycluster/"
OUTPUT_DIR = "../data/parse/yelp/citycluster/"

CNT_DFL = 0
STAR_DFL = 3
DATE_DFL = "2019-12-01 00:00:01"

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

    print("\t[user] loading user interaction ...".format(city))
    with open(INPUT_DIR + "{}/city_user_profile.pkl".format(city), "rb") as fin:
        user_profile = pickle.load(fin)

    user_data = []

    # process users, NOTE: user new index starts with 1
    for user, prof_dict in user_profile.items():
        # --- create feature area ---
        tmp_entry = dict()
        u_elite = prof_dict.get('elite', [])
        tmp_entry['elite_count'] = len(u_elite)  # user elite
        tmp_entry['review_count'] = prof_dict.get('review_count', CNT_DFL)  # review_count
        tmp_entry['fans_count'] = prof_dict.get('fans', CNT_DFL)  # fans
        tmp_entry['funny_score'] = prof_dict.get('funny', CNT_DFL)  # funny
        tmp_entry['cool_score'] = prof_dict.get('cool', CNT_DFL)  # cool
        tmp_entry['useful_score'] = prof_dict.get('useful', CNT_DFL)  # useful
        tmp_entry['avg_stars'] = prof_dict.get('avg_stars', STAR_DFL)  # average stars

        reg_yelp_date = prof_dict.get('yelping_since', DATE_DFL)  # yelping years
        delta_time = datetime.today() - parser.parse(reg_yelp_date)
        tmp_entry['yelping_years'] = delta_time.days // 365
        # --- end create feature area ---

        user_data.append(tmp_entry)

    # create dataframe
    df_user_profile = pd.DataFrame(user_data)

    # non-zero
    df_nonzero = df.fillna(0).astype(bool).sum(axis=1)
    df_nonzero = df_nonzero / len(df_user_profile)
    print("\t[user] non-zero terms in `df_user_profile`")
    print(df_nonzero)

    print("\t[user] saving dataframe to {}".format(OUTPUT_DIR))
    df.to_csv(OUTPUT_DIR + "{}/processed_city_user_profile.csv".format(city),
              index=False)

    return df_nonzero


def extract_business_attr():
    with open(INPUT_DIR + "business_profile.pkl", "rb") as fin:
        business_profile = pickle.load(fin)


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
