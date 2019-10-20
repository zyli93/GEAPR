#! /bin/usr/python

"""
    Yelp dataset preprocessing

    @author: Zeyu Li <zyli@cs.ucla.edu>
"""

try:
    import ujson as json
except:
    import json

try:
    import _pickle as pickle
except:
    import pickle

import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import Counter

from utils import dump_pkl, load_pkl
from sklearn.model_selection import train_test_split

DATA_DIR = "./data/raw/yelp/"
PARSE_DIR = "./data/parse/yelp/"
INTERACTION_DIR = "./data/parse/interactions/"

CANDIDATE_CITY = ['Las Vegas', 'Toronto', 'Phoenix', 'Charlotte']


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
    user_adj = {}

    if not os.path.isdir(PARSE_DIR):
        os.mkdir(PARSE_DIR)

    kept_users = load_pkl(PARSE_DIR + "kept.user.hash")
    kept_users = set(kept_users)

    with open(DATA_DIR + "user.json", "r") as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)
            user_id = data['user_id']
            if user_id not in kept_users:  # discard infrequent or irrelevant cities
                continue
            user_adj[user_id] = data['friends'].split(", ")
            del data['friends']
            del data['user_id']
            user_profile[user_id] = data

    # user adjacency and profile dictionary separately
    print("\tdumping user-friendship and user-profile information ...")
    dump_pkl(PARSE_DIR + "user.friend.pkl", user_adj)
    dump_pkl(PARSE_DIR + "user.profile.pkl", user_profile)


def parse_business():
    """draw business information from business.json

    output:
        business.profile.pkl
        city.business.pkl
    """

    city_business = {}
    business_profiles ={}

    print("[parse_business] parsing all business without selecting cities ...")

    # count business by city and state
    with open(DATA_DIR + "business.json", "r") as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)
            city = data['city']
            business_id = data["business_id"]

            # remove id, state, attributes, and hours
            del data["business_id"], data["state"]
            del data["attributes"], data["hours"]
            business_profiles[business_id] = data

            # save business id to city.business dictionary
            city_business[city] = city_business.get(city, [])
            city_business[city].append(business_id)

    # save city business mapping
    print("[parse business] dumping business.profile and city.business ...")
    dump_pkl(PARSE_DIR + "business.profile.pkl", business_profiles)
    dump_pkl(PARSE_DIR + "city.business.pkl", city_business)


def parse_interactions(keep_city, min_count):
    """draw interact from `review.json` and `tips.json`.

    output: ub.interact.csv

    Args:
        keep_city - the interact of cities to keep
        min_count - min number of instance to keep

    """

    business_profile = load_pkl(PARSE_DIR + "business.profile.pkl")
    keep_city = set(keep_city)

    users = []
    businesses = []
    cities = []

    print("[parse interactions] loading review.json ...")
    with open(DATA_DIR + "review.json", "r") as fin:
        for ln in fin:
            data = json.loads(ln)
            users.append(data['user_id'])
            bid = data['business_id']
            businesses.append(bid)
            cities.append(business_profile[bid]["city"])

    interact = pd.DataFrame({
        'user': users, 'business': businesses, "city": cities})

    # remove duplicate reviews
    print("\tremoving duplicates ...")
    interact.drop_duplicates(
        subset=['user', 'business'], keep="first", inplace=True)

    # select cities
    print("\tfiltering cities from keep_cities...")
    interact = interact[interact['city'].isin(keep_city)]

    # remove appearances less than min-count
    b_counter = Counter(interact.business)
    u_counter = Counter(interact.user)
    interact["b_count"] = interact.business.apply(lambda x: b_counter[x])
    interact["u_count"] = interact.user.apply(lambda x: u_counter[x])

    # filter out entries under `min_count`
    print("removing entries under min_count")
    interact = interact[
        (interact.b_count >= min_count) & (interact.u_count >= min_count)]

    interact.to_csv(PARSE_DIR + "user.business.interact.csv")

    # kept user for parse user
    user_kept = interact["user"].unique().tolist()
    dump_pkl(PARSE_DIR + "kept.user.hash", user_kept)


def city_clustering(city,
                    user_profile,
                    business_profile,
                    business_of_city,
                    interactions,
                    user_friendship):
    """
    narrow down information to specific cities

    Args:
        city - city to study
        user_profile - all user profiles
        business_profile - all business profiles
        business_of_city - list of business in the city
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
    print("\tProcessing city: {}".format(city))

    # make specific folder for city
    city_dir = PARSE_DIR + city + "/"
    if not os.path.isdir(city_dir):
        os.mkdir(city_dir)

    city_user_friendship = {}  # new_id: friends in new_id
    city_user_profile = {}
    city_business_profile = {}

    interaction_of_city = interactions[interactions["city"] == city]

    user_of_city = interaction_of_city['user'].unique().tolist()  # list

    # user, business new id starting from 1 to len(user_of_city)
    city_u2i = dict(zip(user_of_city, range(1, len(user_of_city) + 1)))
    city_b2i = dict(zip(business_of_city, range(1, len(business_of_city) + 1)))

    for uid in user_of_city:
        intersection = np.intersect1d(
            user_of_city, user_friendship[uid], assume_unique=True).tolist()
        city_user_friendship[city_u2i[uid]] = [city_u2i[x] for x in intersection]
        # TODO: check overall user adj contain unique friends

    # make city specific user profile
    for u in user_of_city:
        profile = user_profile[u]
        profile["user_id"] = city_u2i[u]
        city_user_profile[city_u2i[u]] = profile

    # make city specific business profile
    for b in business_of_city:
        profile = business_profile[b]
        profile["business_id"] = city_b2i[b]
        city_business_profile[city_b2i[b]] = profile

    # normalized interactinos
    interaction_of_city['user'] = interaction_of_city['user'].\
        apply(lambda x: city_u2i[x])
    interaction_of_city['business'] = interaction_of_city['business'].\
        apply(lambda x: city_b2i[x])

    # save business_list, user_friendship, and
    dump_pkl(city_dir + "businesses.pkl", business_of_city)
    dump_pkl(city_dir + "users.pkl", user_of_city)
    dump_pkl(city_dir + "business.newid.map", city_b2i)
    dump_pkl(city_dir + "user.newid.map", city_u2i)

    dump_pkl(city_dir + "user.friend.pkl", city_user_friendship)

    dump_pkl(city_dir + "business.profile.pkl", city_business_profile)
    dump_pkl(city_dir + "user.business.profile.pkl", city_user_profile)

    interaction_of_city.to_csv(city_dir + "user.business.interaction.csv")

    print("\tCity {} parsed!".format(city))


def generate_data(city):
    ub = pd.read_csv(PARSE_DIR + city + \
                     "/user.business.interactions.csv")
    users = ub.user.tolist()
    businesses = ub.business.tolist()

    print("\tzipping positive samples ...")
    pos_samples = set(zip(users, businesses))
    pos_count = ub.shape[1]

    neg_samples = []

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

    df_data = pd.concat([df_neg, df_pos], axis=0).reset_index()

    train_df, test_df = train_test_split(
        df_data, random_state=723, test_size=0.1)

    train_df.to_csv(INTERACTION_DIR + "train.csv")
    test_df.to_csv(INTERACTION_DIR + "test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cities", help="city list to keep")
    parser.add_argument("-r", "--raw", action="store_true", help="whether from raw")
    parser.add_argument("-c", "--city_cluster", action="store_true",
                        help="whether do city clustering")
    parser.add_argument("-g", "--gen_data", action="store_true",
                        help="whether generate dataset from u/b interactions")

    parser.add_argument("--min_count", type=int, nargs="?",
                        help="Users/movies have to exceed the min count to be used.")

    args = parser.parse_args()

    if args.raw:
        assert args.min_count, "min_count shouldn't be none!"
        print("[-raw] parsing businesses/interactions/users from scratch ...")

        keep_cities = args.cities.strip().split(",")

        parse_business()
        parse_interactions(keep_city=keep_cities, min_count=args.min_count)
        parse_user()

    if args.city_cluster:
        print("parsing by cities: " + args.cities)
        cities = args.cities.strip().split(",")

        user_profile = load_pkl(PARSE_DIR + "user.profile.pkl")
        business_profile = load_pkl(PARSE_DIR + "business.profile.pkl")
        city_business = load_pkl(PARSE_DIR + "city.business.pkl")

        ub_interactions = pd.read_csv(PARSE_DIR + "ub.interactions.csv")

        user_friendships = load_pkl(PARSE_DIR + "user.friend.pkl")

        for city in args.cities:
            assert args.min_count, "--min_count not given"
            city_clustering(city=city,
                            user_profile=user_profile,
                            business_profile=business_profile,
                            business_of_city=city_business[city],
                            interactions=ub_interactions,
                            user_friendship=user_friendships)

    if args.gen_data:
        print("building implicit graph from cities ...")
        keep_cities = args.cities.strip().split(",")
        for city in keep_cities:
            generate_data(city)



