#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances 

# Read the users file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(r'E:\AI Projects\Recommendation Engine\ml-100k\u.user', sep='|', names=u_cols)

# Read the ratings file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')

# Read the items
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
encoding='latin-1')

#Retrieve the training and testing set
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape

# Get the unique number of users in ratings
n_users = ratings.user_id.unique().shape[0]

# Get the number of unique movies in ratings
n_items = ratings.movie_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_items))

# Iterate each row of the df and assign the ratings of each user to each movie
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)

