#!/usr/bin/env python
# coding: utf-8

# ** In this challenge, your task is to predict a transformed count of hazards or pre-existing damages using a dataset of property information. **
# 
# This will enable Liberty Mutual to more accurately identify high risk homes that require additional examination to confirm their insurability.

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

# load train and test
train = pd.read_csv('../input/liberty/train.csv', index_col=0)
test = pd.read_csv('../input/liberty/test.csv', index_col=0)
sample = pd.read_csv('../input/liberty1/sample_submission.csv', index_col=0)


# In[2]:


print(train.shape)
print(test.shape)


# In[3]:


train.head(5)


# In[4]:


train.info()


# In[5]:


train.apply(lambda x: len(x.unique()))


# In[6]:


test.head(5)


# In[7]:


test.info()


# In[8]:


test.apply(lambda x: len(x.unique()))


# both train, test datasets are made up of categorical values.

# In[9]:


sample.head(5)


# # plan
# 
# algorithm : xgboost
# 
# encoding : label_encoding, DictVectorizer
# 
# submission = pred_1(label_encoding) + pred_2(DictVectorizer)

# # Preprocessing

# In[10]:


labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

train_s = train
test_s = test

columns = train.columns
test_ind = test.index

train_s = np.array(train_s)
test_s = np.array(test_s)


# In[11]:


# label encode the categorical variables
for i in range(train_s.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_s[:, i]) + list(test_s[:, i]))
    train_s[:, i] = lbl.transform(train_s[:, i])
    test_s[:, i] = lbl.transform(test_s[:, i])

train_s = train_s.astype(float)
test_s = test_s.astype(float)


# XGboost Function

# In[12]:


def xgboost_pred(train, labels, test):
    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.005
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.7
    params["scale_pos_weight"] = 1
    params["silent"] = 1
    params["max_depth"] = 9

    plst = list(params.items())

    # Using 4000 rows for early stopping.
    offset = 4000

    num_rounds = 10000
    xgtest = xgb.DMatrix(test)

    # create a train and validation dmatrices
    xgtrain = xgb.DMatrix(train[offset:, :], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset, :], label=labels[:offset])

    # train using early stopping and predict
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

    # reverse train and labels and use different 5k for early stopping.
    train = train[::-1, :]
    labels = np.log(labels[::-1])

    xgtrain = xgb.DMatrix(train[offset:, :], label=labels[offset:])
    xgval = xgb.DMatrix(train[:offset, :], label=labels[:offset])

    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=120)
    preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

    # combine predictions
    preds = (preds1) * 1.4 + (preds2) * 8.6
    return preds


# In[13]:


# model_1. xgboost - label encoding

preds1 = xgboost_pred(train_s, labels, test_s)


# In[14]:


# model_2. xgboost - DictVectorizer

train = train.T.to_dict().values()
test = test.T.to_dict().values()

vec = DictVectorizer()
train = vec.fit_transform(train)
test = vec.transform(test)

preds2 = xgboost_pred(train, labels, test)

preds = 0.47 * (preds1 ** 0.2) + 0.53 * (preds2 ** 0.8)


# In[15]:


# generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('submission.csv')

