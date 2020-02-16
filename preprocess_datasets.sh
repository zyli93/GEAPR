#!/bin/bash

echo "Parsing raw data!"
python preprocess/prep_yelp.py preprocess

echo "Clustering data from the same city!"
python preprocess/prep_yelp.py city_cluster --business_min_count 10 --user_min_count 10

echo "Generating training and testing!"
python preprocess/prep_yelp.py gen_data --train_test_ratio=9:1
