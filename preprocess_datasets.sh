#!/bin/bash

echo "Parsing raw data!"
python preprocess/prep_yelp.py preprocess

echo "Clustering data from the same city!"
python preprocess/prep_yelp.py city_cluster --business_min_count 10 --user_min_count 10

echo "Generating training and testing!"
python preprocess/prep_yelp.py gen_data --train_test_ratio=9:1

echo "Feature Engineering for Toronto"
python preprocess/build_graphs.py --yelp_city=tor --rwr_order=3 --rwr_constant 0.05 --use_sparse_mat=True
python preprocess/attributes_extractor.py tor
python preprocess/geolocations.py --city=tor --num_lat_grid 30 --num_long_grid 30 --num_user 9582 --num_business 9102

echo "Feature Engineering for Phoenix"
python preprocess/build_graphs.py --yelp_city=phx --rwr_order=3 --rwr_constant 0.05 --use_sparse_mat=True
python preprocess/attributes_extractor.py phx
python preprocess/geolocations.py --city=phx --num_lat_grid 30 --num_long_grid 30 --num_user 11289 --num_business 9633
