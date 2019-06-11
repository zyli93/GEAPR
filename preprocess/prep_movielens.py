#! /usr/bin/python3

"""
    File for preprocessing dataset

    @author: Zeyu Li <zyli@cs.ucla.edu>
    
    Note:
        1. `movieId` does not start from 1 but `userId` does.
"""

import os
import sys
import pandas as pd
from collections import Counter
from scipy.sparse import *
import numpy as np

try:
    import _pickle as pickle
except:
    import pickle


RAWDIR = "./data/raw/"
PARSEDIR  = "./data/parse/"

GENRES = ["Action", "Adventure", "Animation",
          "Children", "Comedy", "Crime",
          "Documentary", "Drama", 
          "Fantasy", "Film-Noir", "Horror", "IMAX",
          "Musical", "Mystery", "Romance", 
          "Sci-Fi", "Thriller",
          "War", "Western", "(no genres listed)"]

GENRE_DICT = dict(zip(GENRES, range(len(GENRES))))


def parse_ml():
    """parse movielens

    naming:
        t, i, u, m: tags ,indices, users, movies
        ti: movie titles

    dicts:
        t2i: tag string to tag index
        i2t: tag index to tag string, (also tag set)

        mi2i: movieId to new movie ID starting from 0 [dict]
        i2mi: new movie ID starting from 0 to movieId [list]

        mt: movie tags (mid to list of tag) [dict]
        mg: movie genres (mid to list of genre) [dict]
        mti: movie titles (mid to title string) [dict]
    """

    tag_freq_threshold = 10

    # TODO: dump file before moving forward to next file

    # in_folder = RAWDIR + "ml-20m/"
    in_folder = RAWDIR + "ml-latest-small/"
    out_folder = PARSEDIR + "ml/"
    
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    # Tags dictionary
    tset = set()
    t2i, i2t = {}, []

    # Movies dictionary: tags, genres, title
    mt, mg, mti = {}, {}, {}

    # ================
    #    movies.csv
    # ================

    print("processing movies.csv")

    # build movie profile information
    df_movies = pd.read_csv(in_folder + "movies.csv")

    # i2mi and mi2i
    i2mi = df_movies.movieId.tolist()
    mi2i = dict(zip(i2mi, range(len(i2mi))))

    # parse movie v.s. genre
    for index, row in df_movies.iterrows():
        mid, mtitle, mgenres = row['movieId'], row['title'], row['genres']
        mti[mi2i[mid]] = mtitle
        mg[mi2i[mid]] = [GENRE_DICT[x] for x in mgenres.strip().split("|")]

    # output movie title and movie genre
    print("dumping movies related by pickle")
    with open(out_folder + "mti.pkl", "wb") as fmti,\
        open(out_folder + "mg.pkl", "wb") as fmg:
        pickle.dump(mti, fmti)
        pickle.dump(mg, fmg)

    # dump movieId to new movie id
    with open(out_folder + "i2mi.pkl", "wb") as fi2mi,\
        open(out_folder + "mi2i.pkl", "wb") as fmi2i:
        pickle.dump(i2mi, fi2mi)
        pickle.dump(mi2i, fmi2i)

    # ================
    #    tags.csv
    # ================

    print("processing tags")

    # load csv
    df_tag = pd.read_csv(in_folder + "tags.csv")

    # parse string
    df_tag['tag'] = df_tag['tag'].astype(str)  # convert to string
    df_tag['tag'] = df_tag['tag'].str.replace('[^\w\s]', '')  # remove punc

    # count value, filter to frequent tags
    sr = df_tag['tag'].value_counts()  # count unique value
    i2t = sr[sr > tag_freq_threshold].index.to_list()  # id2tags above bound

    print("Cut-off at {}. {} tags in total"
          .format(tag_freq_threshold, len(i2t)))
    t2i = dict(zip(i2t, range(len(i2t))))

    for index, row in df_tag.iterrows():
        # build movie tag counters
        uid, mid, tag = row['userId'], row['movieId'], row['tag']
        if tag not in t2i:
            continue

        tid, _mid = t2i[tag], mi2i[mid]
        mt[(_mid, tid)] = mt.get((_mid, tid), 0) + 1

    # output of t2i, i2t
    print("dumping tags related by pickle")
    with open(out_folder + "t2i.pkl", "wb") as ft2i,\
        open(out_folder + "i2t.pkl", "wb") as fi2t:
        pickle.dump(t2i, ft2i)
        pickle.dump(i2t, fi2t)

    # output  mt
    with open(out_folder + "mt.pkl", "wb") as fmt:
        pickle.dump(mt, fmt)

    # ================
    #    ratings.csv
    # ================

    print("processing ratings")

    # build rating information
    df_ratings = pd.read_csv(in_folder + "ratings.csv")

    # separate ratings by 3.5, 4.0, 4.5
    rt35 = df_ratings[df_ratings['rating'] >= 3.5]

    # map movieId to new movie id
    rt35['movieId'] = rt35['movieId'].map(mi2i)

    # remove timestamp column
    rt35 = rt35.drop(["timestamp"], axis=1)

    # create 4.0 and 4.5 column
    rt40 = rt35[rt35['rating'] >= 4]
    rt45 = rt40[rt40['rating'] >= 4.5]

    # split train/test dataset
    users = rt40['userId'].value_counts()
    movies = rt40['movieId'].value_counts()


    # create 3.5 points records
    print("dumping ratings 3.5 by pickle")
    rt35.to_csv(out_folder + "rt35.csv")

    # create 4.0 points records
    print("dumping ratings 4.0 by pickle")
    rt40.to_csv(out_folder + "rt40.csv")

    # create 4.5 points records
    print("dumping ratings 4.5 by pickle")
    rt45.to_csv(out_folder + "rt45.csv")

    # create profile for 3.5, 4.0, 4.5 points
    print("creating 3.5, 4.0, and 4.5 user profiles")

    # build user rate movie obj.
    # `movieId` already mapped to new id
    urm35 = rt35.groupby('userId')['movieId'].apply(list)
    urm40 = rt40.groupby('userId')['movieId'].apply(list)
    urm45 = rt45.groupby('userId')['movieId'].apply(list)

    print("dumping 3.5, 4.0, and 4.5 user profiles")
    with open(out_folder + "urm35.pkl", "wb") as f35,\
        open(out_folder + "urm40.pkl", "wb") as f40,\
        open(out_folder + "urm45.pkl", "wb") as f45:
        pickle.dump(urm35, f35)
        pickle.dump(urm40, f40)
        pickle.dump(urm45, f45)
        
    print("movielens process done!")

    print("building sparse matrix")

    # statistics
    n_movies = len(mi2i)
    n_users = max(df_ratings.userId)  # note: user Id start from 1 !!!
    n_tags = len(t2i)
    n_genres = len(GENRES)

    # mt (mid, tid): count
    print("building mt sparse matrix")
    mt_rows, mt_cols = list(zip(*mt.keys()))
    mt_data = list(mt.values())
    mt_mat = csr_matrix((mt_data, (mt_rows, mt_cols)), shape=(n_movies, n_tags))
    
    # mg mid: [genres]
    print("building mg sparse matrix")
    mg_rows, mg_cols = [], [] 
    for key, vlist in mg.items():
        mg_cols += vlist
        mg_rows += [key] * len(vlist)

    mg_mat = csr_matrix((np.ones(len(mg_rows)), (mg_rows, mg_cols)),
                        shape=(n_movies, n_genres))

    # urm index: list
    print("building u sparse matrix")
    u_rows, u_cols = [], []

    # ** score options, 45 to 35/40 ... **
    for ind, vlist in urm45.items():
        u_cols += vlist
        u_rows += [ind] * len(vlist)
    u_mat = csr_matrix((np.ones(len(u_cols)), (u_rows, u_cols)), 
                       shape=(n_users + 1, n_movies))  # userId starts from 1

    print("saving to npz")
    save_npz(out_folder + "mt.npz", mt_mat)
    save_npz(out_folder + "mg.npz", mg_mat)
    save_npz(out_folder + "u.npz", u_mat)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("python3 {} [dataset name]".format(sys.argv[0]))
        sys.exit(1)

    # create directory
    if not os.path.isdir(RAWDIR):
        os.mkdir(RAWDIR)
    if not os.path.isdir(PARSEDIR):
        os.mkdir(PARSEDIR)

    dataset = sys.argv[1]

    # options 
    if dataset == "ml":
        parse_ml()
    else:
        print("Invalid {}".format(dataset))

