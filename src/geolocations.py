"""Extracting Geolocation Features

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

Notes:
    - starting from this doc, we use "POI" and "business"
        interchangeably
"""

import sys

BUS_INPUT = "./data/parse/yelp/citycluster/{}/city_business_profile.pkl"

def business_latlong(city):
    in_file = BUS_INPUT.format(city)




def


if __name__ == "__main__":
    if len(sys.argv) < 1 + 1:
        print("invalid argument")
        sys.exit()

    city = sys.argv[1]

