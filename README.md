# GEAPR: 

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>

## What is GEAPR?
*News*: GEAPR has been accepted by the Applied Science Track of CIKM'21!

GEAPR stands for "**G**raph **E**nhanced **A**ttention network for explainable **P**OI **R**ecommendation".
The major architecture of GEAPR is the following:

![GEAPR Architecture](figures/pipeline.png)

In short, it uses the four different modules to analyze four motivating factors of a POI visit, 
namely structural context, neighbor impact, user attribute, and geolocation.

GEAPR can achieve a great performance shown below.

![Performance](figures/perf.png)

We will show how to process the raw data, run GEAPR, and get results.
Please let us know through EMAIL for any questions! 
Please cite our paper if you used the code (after publish).


## Prepare Data

### Python Requirement packages
Successfully running GEARP requires a few dependent Python packages.
```text
ujson
argparse
pandas==0.25.0
numpy
scikit-learn==0.22.1
tqdm
configparser
```
Run these cmds to install all dependencies by one-click.
```shell script
$ cd path/to/GEAPR
$ pip install -r requirements.txt
```
**MOST IMPORTANTLY**, GEARP is run on TensorFlow 1.14.0.
A GPU-enabled environment is recommended because we have only test run it on GPU machines.

### Download raw dataset
Please download the data set from [here](https://www.yelp.com/dataset).
After download, untar it into `./data/raw/yelp` by the steps below, and ready to the next step!
```shell script
$ cd path/to/GEAPR
$ mkdir -p data/raw/yelp/
$ tar -vxf path/to/yelp_dataset.tar -C data/raw/yelp
```

### Preprocessing

#### 0. All in one script
Run this script for a one-click parsing for Toronto and Phoenix datasets by our default settings.
However, you should run them separately for custom settings. 

```shell script
$ bash preprocess_datasets.sh
```

#### 1. Parse the raw datasets
Just run the cmd below, it doesn't require any dataset-specific arguments.
```bash
$ python preprocess/prep_yelp.py preprocess
```

#### 2. Cluster the data records by cities
Now we separate the dataset by different cities. Set the minimum number of business and user. 
**NOTE** that in the README and source code, we use `business` and `POI` interchangeably.
```bash
$ python preprocess/prep_yelp.py city_cluster --business_min_count [bmc] --user_min_count [umc]
```

For example, if both minimum business count and minimum user count are 10, then we have:
```bash
$ python preprocess/prep_yelp.py city_cluster --business_min_count 10 --user_min_count 10
```
Running this step will generate the statistics of datasets. We summarize them as the following.
`lv` stands for Las Vegas, `tor` stands for Toronto, and `phx` stands for Pheonix.

#### 3. Generate train, test, and validation dataset
Generate train:test dateset, the ratio should be two integers.
```bash
$ python preprocess/prep_yelp.py gen_data --train_test_ratio=[train:test]
```
For example, if we choose to use train:test as 9:1, then we should use:
```bash
$ python preprocess/prep_yelp.py gen_data --train_test_ratio=9:1
```
The statistics for the three datasets.

#### 4. Find the results
Until here, all the datasets are processed. If you are curious about what's inside the processed dataset. Please check below.
In `./data/parse/yelp`, you would be able to see four folders:
* `train_test`: the training set, testing set, and the negative sampling set.
* `citycluster`: all information clustered by cities (`lv`, `tor`, or `phx`)
* `preprocess`: undivided features of preprocessing.
* `interactions`: user-POI interactons 

Among them, `citycluster` and `interactions` will be used in the future procedures.

## Feature Engineering 
Want to run? Not done yet, we need to run some code to extract the features for user and POI such as attributes and POI locations.
Based on the preprocessed features, we further create adjacency matrix features 
and user/POI attribute features. Both of them will be fed into our model.

#### 1. Building Structural Graphes

We are using structural context graphs for later computations. 
Structural context graphs can be generated beforehand.
Here's an example to generate neighbor graphs and structural context graphs:
```bash
$ python preprocess/build_graphs.py --yelp_city=tor --rwr_order=3 --rwr_constant 0.05 --use_sparse_mat=True
```
Here are two tunable hyperparameters:
* `rwr_order`: choose between 2 and 3, number > 3 will generate a much denser graph. Defult is 3.
* `rwr_constant`: rate of re-starting. Default is 0.05.
* `use_sparse`: whether or not to use `scipy.sparse` matrix to save data. Well, the option of `False` has not been tested. Please stick to `True`.


#### 2. Extracting user and item features

We also need to extract features from the user side. This has two steps:

##### 1. Parse attribute features
Go to `./configs/`, there are three examples to set numerical and categorical features in `columns_xx.ini`. The format is the following
```ini
[CATEGORICAL]
col1 = yes  
; `ini` requires an assignment, but the assigned value doesn't matter
col2 = alsoyes 
; ...

[NUMERICAL]
col3 = [#.buckets] ; [#.buckets] should be an integer.
col4 = [#.buckets] ;
```
This tells how many buckets should an attribute col be mapped to.

After setting the `ini` files, just run the following commands to parse attributes:
```bash
$ python preprocess/attributes_extractor.py [city]
```
`city` can be `lv`, `tor`, `phx`, and `all`. `all` will auto run all cities.
This will generate `processed_city_user_profile.csv`, `processed_city_business_profile.csv`, `processed_city_user_profile_distinct.csv`, and `cols_disc_info.pkl`.
in `data/parse/yelp/citycluster/[city]`. 
It will also print out the percentage of empty values under each feature column.
In the later training steps, the model will NOT use `processed_city_user_profile.csv` and `processed_city_business_profile.csv` because: 
(1) we don't use business information in the model; (2) `processed_city_user_profile.csv` isn't parsed to discrete categorical features by bucketing yet.



#### 3. Extract Geolocation features and user/POI adj features
Use the following command to extract the geolocation features and user/POI adjacency features.
```bash
$ python src/geolocations.py --city=[city] --num_lat_grid [n_lat] --num_long_grid [n_long] --num_user [n_user] --num_business [n_poi]
```
For example, in order to handle `tor` dataset, we will input
```bash
$ python src/geolocations.py --city=tor --num_lat_grid 30 --num_long_grid 30 --num_user 9582 --num_business 9102
```
Here's a table telling you how many user/POI/attrs are there (in the default setting).
```text
City    #.user  #.POI  #.attr   
phx     11289   9633   140
tor     9582    9102   140
```

## Run it!
Finally, we are ready to run!

### 1. Running 
Please run this command are an example.
```shell script
$ bash run_yelp.sh
```
Please step into that for the different settings of the parameters.

### 2. Performance
There are two ways of checking the performance. 

1. Check out the standard the output; 
2. Check out `path/to/geapr/output/performance/[trail_id].perf`. It has all the training and testing logs.

### 3. Interpretations
You will need to manually fetch the `output_dict` from the computational graph.
You can find an example in Line 84 of `./geapr/train.py`.


## Appendix

1. To disable the wordy `Warnings` of TensorFlow please add the following:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # << this line disables the warnings
import tensorflow as tf
```
However, the deprecation warnings are not removable.
