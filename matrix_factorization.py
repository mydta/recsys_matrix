#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy import sparse
import math


# In[2]:


anime_ratings_df = pd.read_csv("rating.csv")
anime_ratings_df.shape
print(anime_ratings_df.head())


# In[3]:


anime_ratings = anime_ratings_df.loc[anime_ratings_df.rating != -1].reset_index()[['user_id','anime_id','rating']]
print(anime_ratings.shape)
anime_ratings.head()


# In[4]:


anime_ratings.hist(column = 'rating')


# In[5]:


train_df, valid_df = train_test_split(anime_ratings, test_size=0.2)
train_df = train_df.reset_index()[['user_id', 'anime_id', 'rating']]
valid_df = valid_df.reset_index()[['user_id', 'anime_id', 'rating']]


# In[6]:


# Encodes a pandas column with continous IDs
def encode_column(column):
    keys = column.unique()
    key_to_id = {key:idx for idx,key in enumerate(keys)}
    return key_to_id, np.array([key_to_id[x] for x in column]), len(keys)


# In[7]:


# Encodes rating data with continuous user and anime ids
def encode_df(anime_df):
    anime_ids, anime_df['anime_id'], num_anime = encode_column(anime_df['anime_id'])
    user_ids, anime_df['user_id'], num_users = encode_column(anime_df['user_id'])
    return anime_df, num_users, num_anime, user_ids, anime_ids


# In[8]:


anime_df, num_users, num_anime, user_ids, anime_ids = encode_df(train_df)
print("Number of users :", num_users)
print("Number of anime :", num_anime)
anime_df.head()


# In[9]:


# n: number of items/users
# K: number of factors in the embedding
def create_embeddings(n, K):
    return 11*np.random.random((n, K)) / K


# In[10]:


def create_sparse_matrix(df, rows, cols, column_name = "rating"):
    return sparse.csc_matrix((df[column_name].values,(df['user_id'].values, df['anime_id'].values)),shape=(rows, cols))


# In[11]:


anime_df, num_users, num_anime, user_ids, anime_ids = encode_df(train_df)
Y = create_sparse_matrix(anime_df, num_users, num_anime)


# In[12]:


Y.todense()


# In[13]:


def predict(df, emb_user, emb_anime):
    df['prediction'] = np.sum(np.multiply(emb_anime[df['anime_id']],emb_user[df['user_id']]), axis=1)
    return df


# In[14]:


lmbda = 0.0002


# In[15]:


def cost(df, emb_user, emb_anime):
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_anime.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_anime), emb_user.shape[0], emb_anime.shape[0], 'prediction')
    return math.sqrt(np.sum((Y-predicted).power(2))/df.shape[0])


# In[16]:


def gradient(df, emb_user, emb_anime):
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_anime.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_anime), emb_user.shape[0], emb_anime.shape[0], 'prediction')
    delta =(Y-predicted)
    grad_user = (-2/df.shape[0])*(delta*emb_anime) + 2*lmbda*emb_user
    grad_anime = (-2/df.shape[0])*(delta.T*emb_user) + 2*lmbda*emb_anime
    return grad_user, grad_anime


# In[17]:


# emb_user: the trained user embedding
# emb_anime: the trained anime embedding
def gradient_descent(df, emb_user, emb_anime, iterations=2000, learning_rate=0.01, df_val=None):
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_anime.shape[0])
    beta = 0.9
    grad_user, grad_anime = gradient(df, emb_user, emb_anime)
    v_user = grad_user
    v_anime = grad_anime
    for i in range(iterations):
        grad_user, grad_anime = gradient(df, emb_user, emb_anime)
        v_user = beta*v_user + (1-beta)*grad_user
        v_anime = beta*v_anime + (1-beta)*grad_anime
        emb_user = emb_user - learning_rate*v_user
        emb_anime = emb_anime - learning_rate*v_anime
        if(not (i+1)%50):
            print("\niteration", i+1, ":")
            print("train rmse:",  cost(df, emb_user, emb_anime))
            if df_val is not None:
                print("validation rmse:",  cost(df_val, emb_user, emb_anime))
    return emb_user, emb_anime


# In[ ]:


emb_user = create_embeddings(num_users, 5)
emb_anime = create_embeddings(num_anime, 5)
emb_user, emb_anime = gradient_descent(anime_df, emb_user, emb_anime, iterations=800, learning_rate=1)


# In[ ]:


def encode_new_data(valid_df, user_ids, anime_ids):
    df_val_chosen = valid_df['anime_id'].isin(anime_ids.keys()) & valid_df['user_id'].isin(user_ids.keys())
    valid_df = valid_df[df_val_chosen]
    valid_df['anime_id'] =  np.array([anime_ids[x] for x in valid_df['anime_id']])
    valid_df['user_id'] = np.array([user_ids[x] for x in valid_df['user_id']])
    return valid_df


# In[ ]:


print("before encoding:", valid_df.shape)
valid_df = encode_new_data(valid_df, user_ids, anime_ids)
print("after encoding:", valid_df.shape)


# In[ ]:


train_rmse = cost(train_df, emb_user, emb_anime)
val_rmse = cost(valid_df, emb_user, emb_anime)
print(train_rmse, val_rmse)


# In[ ]:


valid_df.head()

