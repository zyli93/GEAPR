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
        t, i, u, m, ti: tags ,indices, users, movies, movie titles

    dicts:
        t2i: tag string to tag index
        i2t: tag index to tag string, (also tag set)

        mi2i: movieId to new movie ID starting from 0 [dict]
        i2mi: new movie ID starting from 0 to movieId [list]

        mt: movie tags (mid to list of tag) [dict]
        mgr: movie genres (mid to list of genre) [dict]
        mti: movie titles (mid to title string) [dict]
    """

    tag_freq_threshold = 10  # TODO: what is this?

    # TODO: dump file before moving forward to next file

    # in_folder = RAWDIR + "ml-20m/"
    in_folder = RAWDIR + "ml-latest-small/"
    out_folder = PARSEDIR + "movielens/"
    
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    # Movies dictionary: tags, genres, title
    movies_tag, movies_genre, movies_title = {}, {}, {}

    # ================
    #    movies.csv
    # ================

    print("processing movies.csv")
    df_movies = pd.read_csv(in_folder + "movies.csv")
    ind2movie_id = df_movies.movieId.tolist()
    movie_id2ind = dict(zip(ind2movie_id, range(len(ind2movie_id))))

    # parse movie v.s. genre
    for index, row in df_movies.iterrows():
        mid, mtitle, mgenres = row['movieId'], row['title'], row['genres']
        movies_title[movie_id2ind[mid]] = mtitle
        movies_genre[movie_id2ind[mid]] = [GENRE_DICT[x] for x in mgenres.strip().split("|")]

    # output movie title and movie genre
    print("dumping movies related by pickle")
    with open(out_folder + "mti.pkl", "wb") as fmti,\
        open(out_folder + "mgr.pkl", "wb") as fmgr:
        pickle.dump(movies_title, fmti)
        pickle.dump(movies_genre, fmgr)

    # dump movieId to new movie id
    with open(out_folder + "ind2movie_id.pkl", "wb") as fi2mi,\
        open(out_folder + "movie_id2ind.pkl", "wb") as fmi2i:
        pickle.dump(ind2movie_id, fi2mi)
        pickle.dump(movie_id2ind, fmi2i)

    # ================
    #    tags.csv
    # ================

    print("processing tags")
    df_tag = pd.read_csv(in_folder + "tags.csv")

    # parse string
    df_tag['tag'] = df_tag['tag'].astype(str)  # convert to string
    df_tag['tag'] = df_tag['tag'].str.replace('[^\w\s]', '')  # remove punc

    # extract all tags appear greater than `tag_freq_threshold` times.
    sr = df_tag['tag'].value_counts()
    ind2tag = sr[sr > tag_freq_threshold].index.to_list()
    tag2ind = dict(zip(ind2tag, range(len(ind2tag))))
    print("Cut-off tag frequency at {}. {} tags left in total"
          .format(tag_freq_threshold, len(ind2tag)))

    # build movie tag counters
    for index, row in df_tag.iterrows():
        uid, mid, tag = row['userId'], row['movieId'], row['tag']
        if tag not in tag2ind:
            continue

        tag_id, new_mid = tag2ind[tag], movie_id2ind[mid]
        movies_tag[(new_mid, tag_id)] = movies_tag.get((new_mid, tag_id), 0) + 1

    # output of tag2ind, ind2tag
    print("dumping tags related by pickle")
    with open(out_folder + "tag2ind.pkl", "wb") as ft2i,\
        open(out_folder + "ind2tag.pkl", "wb") as fi2t:
        pickle.dump(tag2ind, ft2i)
        pickle.dump(ind2tag, fi2t)

    # output  movies_tag
    with open(out_folder + "movies_tag.pkl" "wb") as fmt:
        pickle.dump(movies_tag, fmt)

    # ================
    #    ratings.csv
    # ================

    print("processing ratings")
    print("considering all ratings as likes")
    df_ratings = pd.read_csv(in_folder + "ratings.csv")
    df_ratings['movie_index'] = df_ratings['movieId'].map(movie_id2ind)
    df_ratings = df_ratings.drop(["timestamp"], axis=1)

    # TODO: save the dataframe
    # TODO: drop the movieId
    # TODO: see if to drop userId
    # TODO: see if userId is from 1 to max

    # build user rate movie matrix
    user_rating = df_ratings.groupby("userId")["movie_index"].apply(list)
    user_rating.to_csv(out_folder + "user_rating.csv", header=True)
    df_ratings.to_csv(out_folder + "rating.csv")

    print("movielens process done!")

    print("building sparse matrix")

    # statistics
    n_movies = len(movie_id2ind)
    n_users = max(df_ratings.userId)  # note: user Id start from 1 !!!
    n_tags = len(tag2ind)
    n_genres = len(GENRES)

    # movies _tag (mid, tid): count
    print("building movies-tags sparse matrix")
    mt_rows, mt_cols = list(zip(*movies_tag.keys()))
    mt_data = list(movies_tag.values())
    mt_mat = csr_matrix((mt_data, (mt_rows, mt_cols)), shape=(n_movies, n_tags))
    
    # movies_genre mid: [genres]
    print("building mgr sparse matrix")
    mg_rows, mg_cols = [], [] 
    for key, vlist in movies_genre.items():
        mg_cols += vlist
        mg_rows += [key] * len(vlist)
    mg_mat = csr_matrix((np.ones(len(mg_rows)), (mg_rows, mg_cols)),
                        shape=(n_movies, n_genres))

    print("building user rate movie sparse matrix")
    u_rows, u_cols = [], []

    for ind, vlist in user_rating.items():
        u_cols += vlist
        u_rows += [ind] * len(vlist)
    u_mat = csr_matrix((np.ones(len(u_cols)), (u_rows, u_cols)), 
                       shape=(n_users + 1, n_movies))  # userId starts from 1

    print("saving to npz")
    save_npz(out_folder + "mt.npz", mt_mat)
    save_npz(out_folder + "mgr.npz", mg_mat)
    save_npz(out_folder + "u.npz", u_mat)


if __name__ == "__main__":
    # create directory
    if not os.path.isdir(RAWDIR):
        os.mkdir(RAWDIR)
    if not os.path.isdir(PARSEDIR):
        os.mkdir(PARSEDIR)
    parse_ml()

