# Interpretable Recommender System with Frienship Networks

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

There's nothing in this one now.

## TODO
1. `src/build_graphes.py` has been changed a lot. Need to filter out unused functions.
2. `src/build_graphes.py` fix all the TODO terms.

## Notes
1. `src/build_graphes.py.old` is a file backuped on Nov.24.

## Preprocessing

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
$ python src/prep_yelp.py gen_data --ttv_ratio=10:1:1
```

#### 4. Find the results
In `./data/parse/yelp`, you would be able to see three folders:
* `preprocess`: undivided features of preprocessing.
* `citycluster`: all information clustered by cities (`lv`, `tor`, or `phx`)
* `interactions`: user-business interacton and synthesized negative samples divided into `training`,
    `testing`, and `validation`.


Among them, `citycluster` and `interactions` will be used in the future procedures.


## Run `dugrilp`


