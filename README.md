# Interpretable Recommender System with Frienship Networks

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

There's nothing in this one now.

## TODO
1. `src/build_graphs.py` has been changed a lot. Need to filter out unused functions.
2. tune hyperparameters from `src/build_graphs.py`: `rwr_order` and `rwr_constant`.
3. `src/main_zyli.py` handle `TODO`.
4. Now that `prep_yelp.py` has been changed, redo the doc of it.
5. Now that `attributes_extractor.py` has been changed, redo the doc of it.
6. Fix all TODO's in `model.py` and `training.py`.
7. Add more features on user from business 
8. 


## Notes
1. `src/build_graphes.py.old` is a file backuped on Nov.24.

## Download raw dataset

## Preprocess raw dataset

### Yelp dataset

#### 1. Run preprocessing on yelp dataset
```bash
$ python src/prep_yelp.py preprocess
```

#### 2. Cluster the data records by cities
```bash
$ python src/prep_yelp.py city_cluster --business_min_count [bmc] --user_min_count [umc]
```

For example, if both minimum business count and minimum user count are 10, then we have:
```bash
$ python src/prep_yelp.py city_cluster --business_min_count 10 --user_min_count 10
```

Running this step will generate the statistics of datasets. We summarize them as the following.
```text
City    B-mc    U-mc    B-count    U-count
lv      10      10      32901      17146
tor     10      10      9360       8942
phx     10      10      10682      9440
```
`lv` stands for Las Vegas, `tor` stands for Toronto, and `phx` stands for Pheonix.

#### 3. Generate train, test, and validation dataset
```bash
$ python src/prep_yelp.py gen_data --train_test_ratio=[train:test]
```
For example, if we choose to use train:test as 9:1, then we should use:
```bash
$ python src/prep_yelp.py gen_data --train_test_ratio=9:1
```

The statistics
```text
City    #.user  #.business  #.attr   
lv      34289   17395       80       
phx
tor
```

For `lv`, train/test business do have over 17224, 13206, 13035 (overlap).

#### 4. Find the results
In `./data/parse/yelp`, you would be able to see three folders:
* `preprocess`: undivided features of preprocessing.
* `citycluster`: all information clustered by cities (`lv`, `tor`, or `phx`)
* `interactions`: user-business interacton and synthesized negative samples divided into `training`,
    `testing`, and `validation`.


Among them, `citycluster` and `interactions` will be used in the future procedures.

## Feature Engineering 
Based on the preprocessed features, we further create adjacency matrix features 
and user/item attribute features. Both of them will be fed into our model.

### 1. Building Structural Graphes

We are using structural context graphs for later computations. 
Structural context graphs can be generated beforehand.
Here's an example to generate neighbor graphs and structural context graphs:
```bash
$ python src/build_graphs.py --dataset=yelp --yelp_city=lv --rwr_order=3 --rwr_constant 0.05 --use_sparse_mat=True
```
Here are two tunable hyperparameters:
* `rwr_order`: choose between 2 and 3, number > 3 will generate a much denser graph. Defult is 3.
* `rwr_constant`: rate of re-starting. Default is 0.05.


### 2. Extracting user and item features

We also need to extract features from the user side. Just run the following commands:
```bash
$ python src/attributes_extractor.py [city] [num_bkl]
```
This will generate `processed_city_user_profile.csv`, `processed_city_business_profile.csv`, `processed_city_user_profile_distinct.csv`, and `cols_disc_info.pkl`.
in `data/parse/yelp/citycluster/[city]`. 
It will also print out the percentage of empty values under each feature column.
In the later training steps, the model will NOT use `processed_city_user_profile.csv` and `processed_city_business_profile.csv` because: (1) we don't use business information in the model; (2) `processed_city_user_profile.csv` isn't parsed to discrete categorical features by bucketing yet.

Please pay attention to the `[num_bkt]`, we are using bucketing for all features by default. If some of the features are categorical and you don't think applying bucketing on that is a good idea. Please add the column name in `configs/user_attr_discrete.txt`: one-line per column name.


## Run it!

### 1. Parameters
 - What are they?
 - What to tune?

### 2. Running 

### 3. Performance

### 4. Find the interpretations


## Appendix

### Counting based hyperparameters

#### `lv`:
```text
user train max - 34389, min - 1
business train max - 17394, min - 1
user test min included
user test max included
business test min included
business test max 17395
#. of fields: 8
        feature useful_score - count 10
        feature yelping_years - count 10
        feature cool_score - count 10
        feature elite_count - count 10
        feature avg_stars - count 10
        feature review_count - count 10
        feature fans_count - count 10
        feature funny_score - count 10
total distinct features: 80
```

#### `tor`:
```text
user train max - 9582, min - 1
business train max - 9102, min - 1
user test min included
user test max included
business test min included
business test max included
#. of fields: 8
        feature avg_stars - count 10
        feature cool_score - count 10
        feature elite_count - count 10
        feature fans_count - count 10
        feature funny_score - count 10
        feature review_count - count 10
        feature useful_score - count 10
        feature yelping_years - count 10
80
```

#### `phx`:
```text
user train max - 11289, min - 1
business train max - 9633, min - 1
user test min included
user test max included
business test min included
business test max included
#. of fields: 8
        feature avg_stars - count 10
        feature cool_score - count 10
        feature elite_count - count 10
        feature fans_count - count 10
        feature funny_score - count 10
        feature review_count - count 10
        feature useful_score - count 10
        feature yelping_years - count 10
80
```


