# dugrilp

Dual Graph Interpretable Link Prediction

There's nothing in this one now.

## Preprocessing

### Yelp dataset

#### 
1. Run preprocessing on yelp dataset
```bash
$ python src/prep_yelp.py preprocess
```

2. Cluster the data records by cities
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

3. Generate train, test, and validation dataset
```bash
$ python src/prep_yelp.py gen_data --ttv_ratio=10:1:1
```

#### Find the results



##Run `dugrilp`


