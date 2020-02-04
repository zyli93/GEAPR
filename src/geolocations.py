"""Extracting Geolocation Features

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

Notes:
    - starting from this doc, we use "POI" and "business"
        interchangeably
"""

import argparse
import pandas as pd
import numpy as np
from utils import load_pkl
from scipy.stats import norm
from scipy.sparse import csr_matrix, save_npz

BUS_INPUT = "./data/parse/yelp/citycluster/{}/city_business_profile.pkl"
GEO_SCORE_OUT = "./data/parse/yelp/citycluster/{}/"
UB_ADJ_INPUT = "./data/parse/yelp/train_test/{}/train_pos.csv"
UB_ADJ_OUTPUT = "./data/parse/yelp/citycluster/{}/city_user_business_adj_mat.npz"


def business_latlong(city, n_lat, n_long):
    """Extract business latitude and longitude information.
    Separating latitude and longitude because city may not be a perfect square.

    Args:
        city -
        n_lat - number of grids for latitude
        n_long - number of grids for longitude
    """
    in_file = BUS_INPUT.format(city)
    bus_profiles = load_pkl(in_file)
    entries = [{"id": 0, "lat": 0, "long": 0}]
    for i, bp in enumerate(bus_profiles):
        assert i == bp['business_id']  # TODO: correct the attribute name
        entries.append({"id": i, "lat": bp['latitude'], "long": bp["longitude"]})

    bus_prof_df = pd.DataFrame(entries)
    max_lat, min_lat = bus_prof_df.latitude.max(), bus_prof_df.latitude.min()
    max_long, min_long = bus_prof_df.longitude.max(), bus_prof_df.longitude.min()
    print("\t[business lat-long] Lat: max-{}, min-{}, delta-{}").format(
        max_lat, min_lat, max_lat - min_lat)
    print("\t[business lat-long] Long: max-{}, min-{}, delta-{}").format(
        max_long, max_long, max_long - min_long)

    avg_lat = bus_prof_df.latitude.iloc[1:].mean()
    avg_long = bus_prof_df.longitude.iloc[1:].mean()
    bus_prof_df.iloc[0].latitude, bus_prof_df.iloc[0].longitude = avg_lat, avg_long

    print("\t[business lat-long] creating bucketing grids ...")
    bus_prof_df = bus_prof_df.assign(
        lat_grid=pd.cut(bus_prof_df.latitude, n_lat, labels=np.arange(n_lat)))
    bus_prof_df = bus_prof_df.assign(
        long_grid=pd.cut(bus_prof_df.longitude, n_long, labels=np.arange(n_long)))

    geo_scores_list = []
    for direction in [bus_prof_df.longitude.to_numpy(), bus_prof_df.latitude.to_numpy()]:
        # (x - y)
        print("\t[business lat-long] computing lat & long scores ...")
        signed_mht_distance = direction.reshape((1, -1)) - direction.reshape((-1, 1))
        norm_signed_mht_distance = signed_mht_distance / np.std(signed_mht_distance)
        scores = norm.pdf(norm_signed_mht_distance)
        geo_scores_list.append(scores)
    geo_score = geo_scores_list[0] + geo_scores_list[1]  # longitude + latitude
    print("\t[business lat-long] processing business latitude and longitude")
    np.savetxt(GEO_SCORE_OUT.format(city)+"business_influence_scores.csv", geo_score,
               fmt="%.10e", delimiter=",")


def user_business_adj(city, n_user, n_business):
    """Build user-business adjacency matrix for three cities.

    Args:
        city -
        n_user, n_business - used to create sparse matrix
    """
    data = pd.read_csv(UB_ADJ_INPUT.format(city))
    uid_list = data.uid.tolist()  # TODO: to correct attribute names
    bid_list = data.bid.tolist()
    ones = np.ones(shape=len(uid_list))
    ub_adj = csr_matrix((ones, (uid_list, bid_list)),
                        shape=(n_user, n_business), dtype=np.float32)
    print("\t[user business adj] saving user - business adjacency matrix ...")
    save_npz(file=UB_ADJ_OUTPUT.format(city), matrix=ub_adj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", help="The city to process")
    parser.add_argument("--num_lat_grid", type=int, nargs="?",
                        help="Number of latitude grid.")
    parser.add_argument("--num_long_grid", type=int, nargs="?",
                        help="Number of longitude grid.")
    parser.add_argument("--num_user", help="Number users.")
    parser.add_argument("--num_business", help="Number businesses.")
    args = parser.parse_args()

    print("[Geolocations] processing business latitude & longitude ...")
    business_latlong(city=args.city,
                     n_lat=args.num_lat_grid, n_long=args.num_long_grid)

    print("[Geolocations] processing user-business adjacency matrix")
    user_business_adj(city=args.city,
                      n_user=args.num_user, n_business=args.num_business)

    print("[Geolocations] work done!")


