#! /bin/usr/python

"""
    Yelp dataset preprocessing

    @author: Zeyu Li <zyli@cs.ucla.edu>
"""

try:
    import ujson as json
except ImportError as e:
    import json

try:
    import _pickle as pickle
except ImportError as e:
    import pickle

import os, sys
import argparse
import pandas as pd
import numpy as np
from collections import Counter

from utils import dump_pkl, load_pkl
from sklearn.model_selection import train_test_split

DATA_DIR = "./data/raw/yelp/"
PARSE_DIR = "./data/parse/yelp/"
INTERACTION_DIR = "./data/parse/interactions/"

CANDIDATE_CITY = ['Las Vegas', 'Toronto', 'Phoenix']
CITY_NAME_ABBR = {"Las Vegas": "lv", "Toronto": "tor", "Phoenix": "phx"}


def load_user_business():
    """helper function that load user and business"""
    user_profile = load_pkl(PARSE_DIR + "user.profile.pkl")
    business_profile = load_pkl(PARSE_DIR + "business.profile.pkl")
    return user_profile, business_profile


def parse_user():
    """Load users

    output: id2user.pkl,
            user2id.pkl,
            user.friend.pkl,
            user.profile.pkl
    """
    print("\tparsing user ...")
    user_profile = {}
    user_friend = {}

    if not os.path.isdir(PARSE_DIR):
        os.mkdir(PARSE_DIR)

    users_list = load_pkl(PARSE_DIR + "users.list.pkl")
    users_list = set(users_list)

    with open(DATA_DIR + "user.json", "r") as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)
            user_id = data['user_id']
            if user_id not in users_list:  # discard infrequent or irrelevant cities
                continue
            user_friend[user_id] = data['friends'].split(", ")
            del data['friends']
            del data['user_id']
            user_profile[user_id] = data

    # user adjacency and profile dictionary separately
    print("\tdumping user-friendship and user-profile information ...")
    dump_pkl(PARSE_DIR + "user.friend.pkl", user_friend)
    dump_pkl(PARSE_DIR + "user.profile.pkl", user_profile)


def parse_business():
    """extract business information from business.json

    output:
        business.profile.pkl
        city.business.pkl
    """

    city_business = {}  # dictionary of city: [business list]
    business_profiles ={}  # dictionary of business profile

    print("[parse_business] preprocessing all business without selecting cities ...")

    # count business by location (city and state)
    with open(DATA_DIR + "business.json", "r") as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)
            city = data['city']

            if city not in CANDIDATE_CITY:  # only use cities
                continue

            business_id = data["business_id"]
            # remove id, state, attributes, and hours
            # rest fields: name, address, postal-code, latitude/longitude
            #              star, review_count, is_open
            del data["business_id"], data["state"]
            del data["attributes"], data["hours"]
            business_profiles[business_id] = data

            # save business id to city_business dictionary
            city_business[city] = city_business.get(city, [])
            city_business[city].append(business_id)

    # save city business mapping
    print("[parse business] dumping business.profile and city.business ...")
    dump_pkl(PARSE_DIR + "business_profile.pkl", business_profiles)
    dump_pkl(PARSE_DIR + "city_business.pkl", city_business)


def parse_interactions(min_count):
    """draw interact from `review.json` and `tips.json`.

    output: ub.interact.csv

    Args:
        keep_city - the interact of cities to keep
        min_count -  (business min count, user min count)
    """

    # business_profile only contains city in Lv, Tor, and Phx
    business_profile = load_pkl(PARSE_DIR + "business_profile.pkl")

    # minimum count of both business and user
    # b_min_count, u_min_count = min_count

    users, businesses, cities = [], [], []

    # create records as (user, business, city) tuple
    print("[parse interactions] loading review.json ...")
    with open(DATA_DIR + "review.json", "r") as fin:
        for ln in fin:
            data = json.loads(ln)
            _bid = data['business_id']
            if _bid not in business_profile:  # only Lv, Tor, and Phx businesses
                continue
            users.append(data['user_id'])
            businesses.append(_bid)
            cities.append(business_profile[_bid]["city"])

    interactions = pd.DataFrame({
        'user': users, 'business': businesses, "city": cities})

    # remove duplicate reviews
    print("\tremoving duplicates ...")
    interactions.drop_duplicates(subset=['user', 'business'], keep="first", inplace=True)

    # remove rear businesses and users appear less than min-count times
    # print("\tremoving entries under min_count b:{}, u:{}".format(b_min_count, u_min_count))
    # b_counter = Counter(interactions.business)
    # u_counter = Counter(interactions.user)
    # interactions["b_count"] = interactions.business.apply(lambda x: b_counter[x])
    # interactions["u_count"] = interactions.user.apply(lambda x: u_counter[x])
    # interactions = interactions[(interactions.b_count >= b_min_count) & (interactions.u_count >= u_min_count)]

    interactions.to_csv(PARSE_DIR + "user.business.interact.csv", index=False)

    # kept user for parse user
    user_remained = interactions["user"].unique().tolist()
    dump_pkl(PARSE_DIR + "users.list.pkl", user_remained)


def city_clustering(city,
                    user_min_count,
                    business_min_count,
                    user_profile,
                    business_profile,
                    interactions,
                    user_friendships):
    """
    narrow down information to specific cities

    Args:
        city - city to work on
        user_profile - all user profiles
        business_profile - all business profiles
        business_of_city - list of business in the city [dict]
        interactions - all interactions
        user_friendship - the users' friendship relations.

    Return:
        business_of_city: list
        user_of_city: list
        city_b2i, city_u2i: dict, reverse relationship
        city_user_frienship: new id, friendship
        city_user_profile: new id-profile
        city_business_profile: new id-profile
        interaction_of_city: csv files with new ids
    """
    print("\t[--city_cluster] Processing city: {}".format(city))

    # make specific folder for city
    city_dir = PARSE_DIR + CITY_NAME_ABBR[city] + "/"
    if not os.path.isdir(city_dir):
        os.mkdir(city_dir)

    city_user_friendship = {}  # new_id: friends in new_id
    city_user_profile = {}
    city_business_profile = {}

    interactions_of_city = interactions[interactions["city"] == city]

    # remove rear businesses and users appear less than min-count times
    print("\tremoving entries under min_count b:{}, u:{}".format(business_min_count, user_min_count))
    b_counter = Counter(interactions_of_city.business)
    u_counter = Counter(interactions_of_city.user)
    interactions_of_city["b_count"] = interactions_of_city.business.apply(lambda x: b_counter[x])
    interactions_of_city["u_count"] = interactions_of_city.user.apply(lambda x: u_counter[x])
    interactions_of_city = interactions_of_city[(interactions_of_city.b_count >= business_min_count) &
                                                (interactions_of_city.u_count >= user_min_count)]

    user_of_city = interactions_of_city['user'].unique().tolist()  # list
    business_of_city = interactions_of_city["business"].unique().tolist()

    # ** before this point: old user/business id; after this point: new user/business index

    # user, business index starting from 1 to len(user_of_city)
    city_uid2ind = dict(zip(user_of_city, range(1, len(user_of_city) + 1)))
    city_bid2ind = dict(zip(business_of_city, range(1, len(business_of_city) + 1)))

    # create city friendships that are in the same city: city_user_friendship
    for uid in user_of_city:
        intersection = np.intersect1d(user_of_city, user_friendships[uid], assume_unique=True).tolist()
        city_user_friendship[city_uid2ind[uid]] = [city_uid2ind[x] for x in intersection]

    # create city specific user profile using new index: city_user_profile
    for uid in user_of_city:
        profile = user_profile[uid]
        profile["user_index"] = city_uid2ind[uid]
        city_user_profile[city_uid2ind[uid]] = profile

    # create city specific business profile using new index: city_business_profile
    for bid in business_of_city:
        profile = business_profile[bid]
        profile["business_id"] = city_bid2ind[bid]
        city_business_profile[city_bid2ind[bid]] = profile

    # user/business id to index in interactions
    # TODO: need to drop u_count or b_count?
    interactions_of_city['user'] = interactions_of_city['user'].apply(lambda x: city_uid2ind[x])
    interactions_of_city['business'] = interactions_of_city['business'].apply(lambda x: city_bid2ind[x])

    # save business_list, user_friendship, and
    dump_pkl(city_dir + "businesses_of_city.pkl", business_of_city)
    dump_pkl(city_dir + "users_of_city.pkl", user_of_city)
    dump_pkl(city_dir + "business.reindex", city_bid2ind)
    dump_pkl(city_dir + "user.reindex", city_uid2ind)

    dump_pkl(city_dir + "city_user_friend.pkl", city_user_friendship)

    dump_pkl(city_dir + "city_business_profile.pkl", city_business_profile)
    dump_pkl(city_dir + "city_user_profile_pkl", city_user_profile)
    interactions_of_city.to_csv(city_dir + "user_business_interaction.csv", index=False)

    print("\tCity {} parsed!".format(city))


def generate_data(city, ratio):
    """
    Create datasets as negative sample

    Arg:
        city: the city to work on

    Store:
        train.csv - training data csv
        test.csv - testing data.csv
        valid.csv - validation data csv
    """
    ub = pd.read_csv(PARSE_DIR + CITY_NAME_ABBR[city] + "/user_business_interactions.csv")
    users = ub.user.tolist()
    businesses = ub.business.tolist()

    print("\tzipping positive samples ...")
    pos_samples = set(zip(users, businesses))
    pos_count = ub.shape[1]

    neg_samples = []

    # Sample positive samples and negative samples
    # TODO: may need to think of better sampling algorithms
    while len(neg_samples) < pos_count:
        sample_u = np.random.choice(users)
        sample_b = np.random.choice(businesses)
        if (sample_u, sample_b) not in pos_samples:
            neg_samples.append((sample_u, sample_b))

    neg_samples = list(zip(*neg_samples))
    df_neg = pd.DataFrame({
        "user": neg_samples[0],
        "business": neg_samples[1],
        "label": 0})

    df_pos = ub[['user'], ['business']]
    df_pos['label'] = 1

    # ratio: Train, Test, Validation
    df_data = pd.concat([df_neg, df_pos], axis=0).reset_index()
    train_df, test_df = train_test_split(df_data, random_state=723, test_size=())
    train_df, valid_df = train_test_split(train_df, random_state=723, test_size=(1/9))

    train_df.to_csv(INTERACTION_DIR + "train.csv", index=False)
    test_df.to_csv(INTERACTION_DIR + "test.csv", index=False)
    valid_df.to_csv(INTERACTION_DIR + "valid.csv", index=False)

    print("[--generate_data] Finished generating datasets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Preprocess business, user-business interaction, and users")
    parser.add_argument("--city_cluster", action="store_true", help="whether do city clustering")
    parser.add_argument("-g", "--gen_data", action="store_true", help="whether generate dataset from u/b interactions")
    parser.add_argument("--business_min_count", type=int, nargs="?", help="Business appearance has to be greater than the min count to be used.")
    parser.add_argument("--user_min_count", type=int, nargs="?", help="User appearance has to be greater than the min count to be used.")
    parser.add_argument("--ttv_ratio", help="Ratio of train, test, and validation sets")
    args = parser.parse_args()

    if args.preprocess:
        assert args.business_min_count, "business_min_count should not be empty"
        assert args.user_min_count, "user_min_count should not be empty"

        print("[--preprocess] parsing businesses/interactions/users from scratch ...")
        parse_business()
        parse_interactions(min_count=(args.business_min_count, args.user_min_count))
        parse_user()
        print("[--preprocess] done!")

    if args.city_cluster:
        # TODO: add more details in this docstring
        """
        city cluster create city-specific datasets
        - User and business ids are replaced to new ones
        - Friendships are filtered to users only in the same city
        - TODO
        """
        user_profile = load_pkl(PARSE_DIR + "user.profile.pkl")
        business_profile = load_pkl(PARSE_DIR + "business.profile.pkl")
        city_business = load_pkl(PARSE_DIR + "city.business.pkl")
        ub_interactions = pd.read_csv(PARSE_DIR + "ub.interactions.csv")
        user_friendships = load_pkl(PARSE_DIR + "user.friend.pkl")

        for city in CANDIDATE_CITY:
            assert args.min_count, "--min_count not given"  # TODO: what is min count here???
            print("Running City clustering on " + city)
            city_clustering(city=city,
                            user_min_count=args.user_min_count,
                            business_min_count=args.business_min_count,
                            user_profile=user_profile,
                            business_profile=business_profile,
                            interactions=ub_interactions,
                            user_friendships=user_friendships)

    if args.gen_data:
        # TODO: what is gen_data?
        assert args.ttv_ratio, "Train/Test/Validation ratio should not be empty!"
        ttv_ratio = tuple([int(x) for x in args.ttv_ratio.split(":")])
        print("building implicit graph from cities ...")
        for city in CANDIDATE_CITY:
            generate_data(city, ttv_ratio)



