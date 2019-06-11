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

from utils import dump_pkl, load_pkl

DATA_DIR = "./data/raw/yelp/"
PARSE_DIR = "./data/parse/yelp/"

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
    user_dict = {}
    user_adj = {}
    id2user = []

    with open(DATA_DIR + "user.json", "r") as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)
            id2user.append(data['user_id'])
            data['user_id'] = ind
            user_dict[ind] = data

    print("Total users in Raw file: {}".format(len(id2user)))

    # map long id to short id
    user2id = dict(zip(id2user, range(len(id2user))))

    # save id2user, user2id
    dump_pkl(PARSE_DIR + "id2user.pkl", id2user)
    dump_pkl(PARSE_DIR + "user2id.pkl", user2id)

    # separate user friendship and profile information
    for key, vdict in user_dict.items():
        friends_str = vdict['friends']
        friends_list = [user2id[x] for x in friends_str.split(", ")]
        user_adj[key] = friends_list
        del vdict['friends']

    # user adjacency and profile dictionary separately
    dump_pkl(PARSE_DIR + "user.friend.pkl", user_adj)
    dump_pkl(PARSE_DIR + "user.profile.pkl", user_dict)


def parse_business():
    """draw business information from business.json

    output: business2id.pkl,
            id2business.pkl,
            business.profile.pkl
            city.business.pkl
    """

    business_dict = {}
    id2business = []
    city_business = {}

    # count business by city and state
    with open(DATA_DIR + "business.json", "r") as fin:
        for ind, ln in enumerate(fin.readlines()):
            data = json.loads(ln)
            id2business.append(data['business_id'])
            data['business_id'] = ind
            business_dict[ind] = data

            city = data.get('city', "NoCity")  # insert business to city-bus list
            city_business_list = city_business.get(city, [])
            city_business[city] = city_business_list.append(ind)

    # map long id to short id
    business2id = dict(zip(id2business, range(len(id2business))))

    # save bus2id, id2bus, and business profile
    dump_pkl(PARSE_DIR + "business2id.pkl", business2id)
    dump_pkl(PARSE_DIR + "id2business.pkl", id2business)
    dump_pkl(PARSE_DIR + "business.profile.pkl", business_dict)

    # save city business mapping
    dump_pkl(PARSE_DIR + "city.business.pkl", city_business)


def parse_interactions():
    """draw interactions from `review.json` and `tips.json`.

    output: ub.interactions.csv

    TODO: tip and review could be duplicated!
    """
    u2id = load_pkl(PARSE_DIR + "user2id.pkl")
    b2id = load_pkl(PARSE_DIR + "business2id.pkl")

    print("\tloading review.json ...")
    with open(PARSE_DIR + "ub.interactions.csv", "w") as fout:
        print("user_id,business_id,type", file=fout)
        with open(DATA_DIR + "review.json", "r") as fin:
            for ln in fin:
                data = json.loads(ln)
                uid = u2id[data['user_id']]
                bid = b2id[data['business_id']]
                print("{},{},{}".format(uid, bid, "rev"), file=fout)

        with open(DATA_DIR + "tip.json", "r") as fin:
            for ln in fin:
                data = json.loads(ln)
                uid = u2id[data['user_id']]
                bid = b2id[data['business_id']]
                print("{},{},{}".format(uid, bid, "tip"), file=fout)


def city_clustering(city, buslist,
                    interactions,
                    user_friendship):
    """
    narrow down information to specific cities

    Args:
        city - city to study
        buslist - the list of businesses in this specific city
        interactions - all interactions
        user_friendship - the users' friendship relations.
    """
    print("\tProcessing city: {}".format(city))
    city_user_friendship = {}

    # make specific folder
    city_dir = PARSE_DIR + city + "/"
    if not os.path.isdir(city_dir):
        os.mkdir(city_dir)

    city_interaction = interactions[interactions['bid'].isin(buslist)]
    city_users = city_interaction['uid'].unique()  # 1d np array
    for uid in city_users:
        city_user_friendship[uid] = np.intersect1d(
            city_users, user_friendship[uid], assume_unique=True).tolist()
        # TODO: check overall user adj contain unique friends

    # save business_list, user_friendship, and
    dump_pkl(city_dir + "business.list.pkl", buslist)
    dump_pkl(city_dir + "user.friend.pkl", city_user_friendship)
    city_interaction.to_csv(city_dir + "user.business.interaction.csv")

    print("\tCity {} parsed!".format(city))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--raw", action="store_true", help="from raw")
    parser.add_argument("-c", "--cities", help="city names", nargs="?")

    args = parser.parse_args()

    if args.raw:
        print("parsing user/business/interactions from scratch ...")
        parse_user()
        parse_business()
        parse_interactions()

    if args.cities:
        print("parsing by cities: " + args.cities)
        cities = args.cities.strip().split(",")

        # load all city-businesses mapping
        city_bus = load_pkl(PARSE_DIR + "city.business.pkl")

        # load all user friendships
        user_friendships = load_pkl(PARSE_DIR + "user.friend.pkl")

        # load all interactions
        ub_interactions = pd.read_csv(PARSE_DIR + "ub.interactions.csv")

        for city in cities:
            city_clustering(city=city, buslist=city_bus[city],
                            interactions=ub_interactions,
                            user_friendship=user_friendships)




