#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os


# # Exploratory Data Analysis

# In[22]:


# Loading data directly from CatBoost
from catboost.datasets import amazon

train, test = amazon()


# In[23]:


print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))


# In[24]:


train.head(5)


# In[25]:


test.head(5)


# dataset has 9 columns, plus target (`ACTION`) for train and `id` for test. 
# All these columns are categorical encoded as integers.

# # Feature Engineering

# In[26]:


train.apply(lambda x: len(x.unique()))


# `RESOURCE`,`MGR_ID` and `ROLE_FAMILY_DESC`. 
# These 3 columns are high-cardinality categorical features.
# 
# `ROLE_CODE` and `ROLE_TITLE`. 
# These 2 columns have exactly the same amount of unique values.

# In[27]:


import itertools
target = "ACTION"
col4train = [x for x in train.columns if x!=target]

col1 = 'ROLE_CODE'
col2 = 'ROLE_TITLE'

pair = len(train.groupby([col1,col2]).size())
single = len(train.groupby([col1]).size())

print(col1, col2, pair, single)


# these 2 columns have 1:1 relationship.
# remove `ROLE_TITLE`.
# 

# In[28]:


col4train = [x for x in col4train if x!='ROLE_TITLE']
y = train[target].values


# # Encoding

# # Unsupervised categorical encodings

# Label Encoding, SVD Encoding, Frequency encoding

# functions

# In[29]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score

def get_model(): 
    params = {
        "n_estimators":300, 
        "n_jobs": 3,
        "random_state":5436,
    }
    return ExtraTreesClassifier(**params)

def validate_model(model, data):
    skf = StratifiedKFold(n_splits=5, random_state = 4141, shuffle = True)
    stats = cross_validate(
        model, data[0], data[1], 
        groups=None, scoring='roc_auc', 
        cv=skf, n_jobs=None, return_train_score = True
    )
    stats = pd.DataFrame(stats)
    return stats.describe().transpose()

def transform_dataset(train, test, func, func_params = {}):
    dataset = pd.concat([train, test], ignore_index = True)
    dataset = func(dataset, **func_params)
    if isinstance(dataset, pd.DataFrame):
        new_train = dataset.iloc[:train.shape[0],:].reset_index(drop = True)
        new_test =  dataset.iloc[train.shape[0]:,:].reset_index(drop = True)
    else:
        new_train = dataset[:train.shape[0]]
        new_test =  dataset[train.shape[0]:]
    return new_train, new_test


# > ## 1. Label Encoding

# In[30]:


#for each column in dataset creates N column with random integers
def assign_rnd_integer(dataset, number_of_times = 5, seed = 23):
    new_dataset = pd.DataFrame()
    np.random.seed(seed)
    for c in dataset.columns:
        for i in range(number_of_times):
            col_name = c+"_"+str(i)
            unique_vals = dataset[c].unique()
            labels = np.array(list(range(len(unique_vals))))
            np.random.shuffle(labels)
            mapping = pd.DataFrame({c: unique_vals, col_name: labels})
            new_dataset[col_name] = (dataset[[c]]
                                     .merge(mapping, on = c, how = 'left')[col_name]
                                    ).values
    return new_dataset


# In[31]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    assign_rnd_integer, {"number_of_times":5}
)

print(new_train.shape, new_test.shape)
new_train.head(5)


# In[32]:


validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# In[33]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    assign_rnd_integer, {"number_of_times":10}
)
print(new_train.shape, new_test.shape)
new_train.head(5)


# In[34]:


validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# when 5 times : 0.8634
# 
# when 10 times : 0.8728
# 
# choose 10 times

# ## 2. SVD encoding

# In[35]:


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def extract_col_interaction(dataset, col1, col2, tfidf = True):
    data = dataset.groupby([col1])[col2].agg(lambda x: " ".join(list([str(y) for y in x])))
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "))
    else:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "))
    
    data_X = vectorizer.fit_transform(data)
    dim_red = TruncatedSVD(n_components=1, random_state = 5115)
    data_X = dim_red.fit_transform(data_X)
    
    result = pd.DataFrame()
    result[col1] = data.index.values
    result[col1+"_{}_svd".format(col2)] = data_X.ravel()
    return result

import itertools
def get_col_interactions_svd(dataset, tfidf = True):
    new_dataset = pd.DataFrame()
    for col1,col2 in itertools.permutations(dataset.columns, 2):
        data = extract_col_interaction(dataset, col1,col2, tfidf)
        col_name = [x for x in data.columns if "svd" in x][0]
        new_dataset[col_name] = dataset[[col1]].merge(data, on = col1, how = 'left')[col_name]
    return new_dataset


# In[36]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    get_col_interactions_svd
)
print(new_train.shape, new_test.shape)
new_train.head(5)


# In[37]:


validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# SVD : AUC is 0.8625.

# ## 3. Frequency encoding

# In[38]:


def get_freq_encoding(dataset):
    new_dataset = pd.DataFrame()
    for c in dataset.columns:
        data = dataset.groupby([c]).size().reset_index()
        new_dataset[c+"_freq"] = dataset[[c]].merge(data, on = c, how = "left")[0]
    return new_dataset


# In[39]:


new_train, new_test = transform_dataset(
    train[col4train], test[col4train], 
    get_freq_encoding
)
print(new_train.shape, new_test.shape)
new_train.head(5)


# In[40]:


validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# Frequency : AUC is 0.8209

# In[41]:


new_train1, new_test1 = transform_dataset(
    train[col4train], test[col4train], get_freq_encoding
)
new_train2, new_test2 = transform_dataset(
    train[col4train], test[col4train], get_col_interactions_svd
)
new_train3, new_test3 = transform_dataset(
    train[col4train], test[col4train], 
    assign_rnd_integer, {"number_of_times":10}
)

new_train = pd.concat([new_train1, new_train2, new_train3], axis = 1)
new_test = pd.concat([new_test1, new_test2, new_test3], axis = 1)
print(new_train.shape, new_test.shape)
validate_model(
    model = get_model(), 
    data = [new_train.values, y]
)


# 1. Label + SVD + Frequency :  AUC is 0.8799

# # Supervised categorical encodings

# target encoding smoothing adding noise, expanding mean

# function

# 1. simple target encoding

# In[42]:


from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, columns_names ):
        self.columns_names = columns_names
        self.learned_values = {}
        self.dataset_mean = np.nan
    
    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        self.learned_values = {}
        X_["__target__"] = y
        for c in [x for x in X_.columns if x in self.columns_names]:
            self.learned_values[c] = (X_[[c,"__target__"]]
                                      .groupby(c)["__target__"].mean()
                                      .reset_index())
        self.dataset_mean = np.mean(y)
        return self
    
    def transform(self, X, **fit_params):
        transformed_X = X[self.columns_names].copy()
        for c in transformed_X.columns:
            transformed_X[c] = (transformed_X[[c]]
                                .merge(self.learned_values[c], on = c, how = 'left')
                               )["__target__"]
        transformed_X = transformed_X.fillna(self.dataset_mean)
        return transformed_X
    
    def fit_transform(self, X, y, **fit_params):
        self.fit(X,y)
        return self.transform(X)


# In[43]:


skf = StratifiedKFold(n_splits=5, random_state = 5451, shuffle = True)
te = TargetEncoding(columns_names=col4train)
X_tr = te.fit_transform(train, y).values

scores = []
tr_scores = []
for train_index, test_index in skf.split(train, y):
    train_df, valid_df = X_tr[train_index], X_tr[test_index]
    train_y, valid_y = y[train_index], y[test_index]

    model = get_model()
    model.fit(train_df,train_y)

    predictions = model.predict_proba(valid_df)[:,1]
    scores.append(roc_auc_score(valid_y, predictions))

    train_preds = model.predict_proba(train_df)[:,1]
    tr_scores.append(roc_auc_score(train_y, train_preds))

print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(
    np.mean(tr_scores), np.mean(scores), np.std(scores)
))


# Overfitting! solution : adding noise

# ### 2. target encoding smoothing

# In[44]:


class TargetEncodingSmoothing(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns_names,k, f ):
        self.columns_names = columns_names
        self.learned_values = {}
        self.dataset_mean = np.nan
        self.k = k #
        self.f = f #
        
    def smoothing_func(self, N): #
        return 1 / (1 + np.exp(-(N-self.k)/self.f))
    
    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        self.learned_values = {}
        self.dataset_mean = np.mean(y)
        X_["__target__"] = y
        for c in [x for x in X_.columns if x in self.columns_names]:
            stats = (X_[[c,"__target__"]]
                     .groupby(c)["__target__"].
                     agg(['mean', 'size'])) 
            stats["alpha"] = self.smoothing_func(stats["size"])
            stats["__target__"] = (stats["alpha"]*stats["mean"] 
                                   + (1-stats["alpha"])*self.dataset_mean)
            stats = (stats
                     .drop([x for x in stats.columns if x not in ["__target__",c]], axis = 1)
                     .reset_index())
            self.learned_values[c] = stats
        self.dataset_mean = np.mean(y)
        
        return self
    
    def transform(self, X, **fit_params):
        transformed_X = X[self.columns_names].copy()
        for c in transformed_X.columns:
            transformed_X[c] = (transformed_X[[c]]
                                .merge(self.learned_values[c], on = c, how = 'left')
                               )["__target__"]
        transformed_X = transformed_X.fillna(self.dataset_mean)
        
        return transformed_X
    def fit_transform(self, X, y, **fit_params):
        self.fit(X,y)
        return self.transform(X)


# In[45]:


skf = StratifiedKFold(n_splits=5, random_state = 5451, shuffle = True)
scores = []
tr_scores = []
for train_index, test_index in skf.split(train, y):
    train_df = train.loc[train_index,col4train].reset_index(drop = True)
    valid_df = train.loc[test_index,col4train].reset_index(drop = True)
    train_y, valid_y = y[train_index], y[test_index]
    te = TargetEncodingSmoothing(
        columns_names= col4train,
        k = 3, f = 1.5
    )
    
    X_tr = te.fit_transform(train_df, train_y).values
    ##
    cols_mean_target_smoothing = te.fit_transform(train_df, train_y)
    ##
    X_val = te.transform(valid_df).values

    model = get_model()
    model.fit(X_tr,train_y)

    predictions = model.predict_proba(X_val)[:,1]
    scores.append(roc_auc_score(valid_y, predictions))

    train_preds = model.predict_proba(X_tr)[:,1]
    tr_scores.append(roc_auc_score(train_y, train_preds))


# In[46]:


cols_mean_target_smoothing.head(5)


# In[47]:


print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(
    np.mean(tr_scores), np.mean(scores), np.std(scores)
))


# AUC score: 0.7878. 
# 
# ## Adding noise. CV inside CV.
# 
# adding noise : make our embedding noisy
# 
# we split our train dataset into n folds, and we use n-1 folds to create target mean embedding and use it for the last n-th fold.
# 
# 
# Here is the function which does that:

# In[48]:


def get_CV_target_encoding(data, y, encoder, cv = 5):
    skfTE = StratifiedKFold(n_splits=cv, random_state = 545167, shuffle = True)
    result = []
    for train_indexTE, test_indexTE in skfTE.split(data, y):
        encoder.fit(data.iloc[train_indexTE,:].reset_index(drop = True), y[train_indexTE])
        tmp =  encoder.transform(data.iloc[test_indexTE,:].reset_index(drop = True))
        tmp["index"] = test_indexTE
        result.append(tmp)
        
    result = pd.concat(result, ignore_index = True)
    result = result.sort_values('index').reset_index(drop = True).drop('index', axis = 1)
    return result


# In[49]:


scores = []
tr_scores = []
for train_index, test_index in skf.split(train, y):
    train_df = train.loc[train_index,col4train].reset_index(drop = True)
    valid_df = train.loc[test_index,col4train].reset_index(drop = True)
    train_y, valid_y = y[train_index], y[test_index]
    te = TargetEncodingSmoothing(
        columns_names= col4train,
        k = 3, f = 1.5
    )
    
    X_tr = get_CV_target_encoding(train_df, train_y, te, cv = 5)

    te.fit(train_df, train_y)
    X_val = te.transform(valid_df).values

    model = get_model()
    model.fit(X_tr,train_y)

    predictions = model.predict_proba(X_val)[:,1]
    scores.append(roc_auc_score(valid_y, predictions))

    train_preds = model.predict_proba(X_tr)[:,1]
    tr_scores.append(roc_auc_score(train_y, train_preds))

print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(
    np.mean(tr_scores), np.mean(scores), np.std(scores)
))


# AUC : From .78 to .85.

# In[50]:


X_tr.head(5)


# ## Adding noise. Expanding mean.
# 
# Imagine algorithm rolling trough data and for each new row it uses all previously seen rows to calculate this new row mean. For the very first row there is no previously seen rows available so it's mean will be dataset mean. For the second row you can use first (and only first) row, because you already saw it.

# In[51]:


class TargetEncodingExpandingMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns_names):
        self.columns_names = columns_names
        self.learned_values = {}
        self.dataset_mean = np.nan
    def fit(self, X, y, **fit_params):
        X_ = X.copy()
        self.learned_values = {}
        self.dataset_mean = np.mean(y)
        X_["__target__"] = y
        for c in [x for x in X_.columns if x in self.columns_names]:
            stats = (X_[[c,"__target__"]]
                     .groupby(c)["__target__"]
                     .agg(['mean', 'size'])) #
            stats["__target__"] = stats["mean"]
            stats = (stats
                     .drop([x for x in stats.columns if x not in ["__target__",c]], axis = 1)
                     .reset_index())
            self.learned_values[c] = stats
        return self
    def transform(self, X, **fit_params):
        transformed_X = X[self.columns_names].copy()
        for c in transformed_X.columns:
            transformed_X[c] = (transformed_X[[c]]
                                .merge(self.learned_values[c], on = c, how = 'left')
                               )["__target__"]
        transformed_X = transformed_X.fillna(self.dataset_mean)
        return transformed_X
    
    def fit_transform(self, X, y, **fit_params):
        self.fit(X,y)
    
        #Expanding mean transform
        X_ = X[self.columns_names].copy().reset_index(drop = True)
        X_["__target__"] = y
        X_["index"] = X_.index
        X_transformed = pd.DataFrame()
        for c in self.columns_names:
            X_shuffled = X_[[c,"__target__", "index"]].copy()
            X_shuffled = X_shuffled.sample(n = len(X_shuffled),replace=False)
            X_shuffled["cnt"] = 1
            X_shuffled["cumsum"] = (X_shuffled
                                    .groupby(c,sort=False)['__target__']
                                    .apply(lambda x : x.shift().cumsum()))
            X_shuffled["cumcnt"] = (X_shuffled
                                    .groupby(c,sort=False)['cnt']
                                    .apply(lambda x : x.shift().cumsum()))
            X_shuffled["encoded"] = X_shuffled["cumsum"] / X_shuffled["cumcnt"]
            X_shuffled["encoded"] = X_shuffled["encoded"].fillna(self.dataset_mean)
            X_transformed[c] = X_shuffled.sort_values("index")["encoded"].values
        return X_transformed


# In[52]:


scores = []
tr_scores = []
for train_index, test_index in skf.split(train, y):
    train_df = train.loc[train_index,col4train].reset_index(drop = True)
    valid_df = train.loc[test_index,col4train].reset_index(drop = True)
    train_y, valid_y = y[train_index], y[test_index]
    te = TargetEncodingExpandingMean(columns_names=col4train)

    X_tr = te.fit_transform(train_df, train_y)
    X_val = te.transform(valid_df).values

    model = get_model()
    model.fit(X_tr,train_y)

    predictions = model.predict_proba(X_val)[:,1]
    scores.append(roc_auc_score(valid_y, predictions))

    train_preds = model.predict_proba(X_tr)[:,1]
    tr_scores.append(roc_auc_score(train_y, train_preds))

print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(
    np.mean(tr_scores), np.mean(scores), np.std(scores)
))


# AUC score: 0.8694

# In[53]:


X_tr.head(5)


# use feature pairs to create a new set of categorical features. 
# 
# take pair of existing features and concat them together

# In[54]:


train[col4train] = train[col4train].values.astype(str)
test[col4train] = test[col4train].values.astype(str)

from itertools import combinations
new_col4train = col4train
for c1,c2 in combinations(col4train, 2):
    name = "{}_{}".format(c1,c2)
    new_col4train.append(name)
    train[name] = train[c1] + "_" + train[c2]
    test[name] = test[c1] + "_" + test[c2]


# In[55]:


print(train[new_col4train].shape, test[new_col4train].shape)
train[new_col4train].head(5)


# In[56]:


train[new_col4train].apply(lambda x: len(x.unique()))


# > use both `TargetEncodingExpandingMean` and `TargetEncodingSmoothing` with CV to create embeddings.

# In[57]:


scores = []
tr_scores = []
for train_index, test_index in skf.split(train, y):
    train_df = train.loc[train_index,new_col4train].reset_index(drop = True)
    valid_df = train.loc[test_index,new_col4train].reset_index(drop = True)
    train_y, valid_y = y[train_index], y[test_index]
    te = TargetEncodingExpandingMean(columns_names=new_col4train)

    X_tr = te.fit_transform(train_df, train_y)
    X_val = te.transform(valid_df)
    
    te2 = TargetEncodingSmoothing(
        columns_names= new_col4train,
        k = 3, f = 1.5,
    )
    
    X_tr2 = get_CV_target_encoding(train_df, train_y, te2, cv = 5)
    te2.fit(train_df, train_y)
    X_val2 = te2.transform(valid_df)
    
    X_tr = pd.concat([X_tr, X_tr2], axis = 1)
    X_val = pd.concat([X_val, X_val2], axis = 1)

    model = get_model()
    model.fit(X_tr,train_y)

    predictions = model.predict_proba(X_val)[:,1]
    scores.append(roc_auc_score(valid_y, predictions))

    train_preds = model.predict_proba(X_tr)[:,1]
    tr_scores.append(roc_auc_score(train_y, train_preds))

print("Train AUC score: {:.4f} Valid AUC score: {:.4f}, STD: {:.4f}".format(
    np.mean(tr_scores), np.mean(scores), np.std(scores)
))


# > AUC score is 0.8772

# Let's define all things we need. 
# 1. Using original features and 2-nd level interactions (pairs)
# 2. And we will need: Frequency Encoding, Label Encoding, SVD encoding and target encoding.
# 3. we are building 2 datasets and train 2 models. The final submission is an averages of these 2 models.

# In[58]:


#dataset #1
cols_svd = ['MGR_ID_ROLE_CODE','MGR_ID_ROLE_DEPTNAME','MGR_ID_ROLE_FAMILY', 
            'RESOURCE_MGR_ID','RESOURCE_ROLE_CODE', 'RESOURCE_ROLE_FAMILY',
            'RESOURCE_ROLE_ROLLUP_1','RESOURCE_ROLE_ROLLUP_2','RESOURCE',
            'ROLE_DEPTNAME_ROLE_CODE','ROLE_DEPTNAME_ROLE_FAMILY',
            'ROLE_FAMILY_DESC_ROLE_FAMILY','ROLE_FAMILY_ROLE_CODE',
            'ROLE_FAMILY','ROLE_ROLLUP_1_ROLE_DEPTNAME',
            'ROLE_ROLLUP_1_ROLE_FAMILY_DESC', 'ROLE_ROLLUP_1_ROLE_FAMILY',
            'ROLE_ROLLUP_1','ROLE_ROLLUP_2']

cols_rnd = ['MGR_ID_ROLE_DEPTNAME','MGR_ID_ROLE_FAMILY','MGR_ID_ROLE_ROLLUP_1',
 'MGR_ID_ROLE_ROLLUP_2','MGR_ID','RESOURCE_MGR_ID','RESOURCE_ROLE_CODE',
 'RESOURCE_ROLE_FAMILY_DESC','RESOURCE_ROLE_FAMILY','RESOURCE_ROLE_ROLLUP_1',
 'RESOURCE_ROLE_ROLLUP_2','ROLE_DEPTNAME_ROLE_FAMILY_DESC','ROLE_FAMILY_DESC_ROLE_CODE',
 'ROLE_FAMILY_DESC_ROLE_FAMILY','ROLE_FAMILY','ROLE_ROLLUP_1_ROLE_CODE',
 'ROLE_ROLLUP_1_ROLE_DEPTNAME','ROLE_ROLLUP_1_ROLE_FAMILY_DESC','ROLE_ROLLUP_2_ROLE_FAMILY']

cols_freq = ['MGR_ID_ROLE_DEPTNAME','RESOURCE_MGR_ID','RESOURCE_ROLE_CODE',
 'RESOURCE_ROLE_DEPTNAME','RESOURCE_ROLE_FAMILY_DESC','RESOURCE_ROLE_FAMILY',
 'RESOURCE_ROLE_ROLLUP_1','ROLE_DEPTNAME_ROLE_FAMILY_DESC','ROLE_DEPTNAME_ROLE_FAMILY',
 'ROLE_DEPTNAME','ROLE_FAMILY_DESC_ROLE_CODE','ROLE_FAMILY_DESC_ROLE_FAMILY',
 'ROLE_ROLLUP_1_ROLE_CODE','ROLE_ROLLUP_2_ROLE_DEPTNAME']

data_svd = transform_dataset(train[cols_svd], test[cols_svd], get_col_interactions_svd)
data_rnd = transform_dataset(train[cols_rnd], test[cols_rnd], 
                             assign_rnd_integer, {"number_of_times":10})
data_freq = transform_dataset(train[cols_freq], test[cols_freq], get_freq_encoding)


# In[59]:


data_train = pd.concat([x[0] for x in [data_svd, data_rnd, data_freq]], axis = 1)
data_test = pd.concat([x[1] for x in [data_svd, data_rnd, data_freq]], axis = 1)


# In[60]:


print("Dataset shape, Train: {}, Test: {}".format(data_train.shape, data_test.shape))


# In[61]:


validate_model(
    model = get_model(), 
    data = [data_train.values, y]
)


# dataset1 ( Label + SVD + Frequency ) : 0.8905

# In[62]:


del([data_svd, data_rnd, data_freq])
gc.collect()
data_train = data_train.values
data_test = data_test.values
gc.collect()


# In[63]:


model = get_model()
model.fit(data_train, y)
predictions_1 = model.predict_proba(data_test)[:,1]


# In[64]:


del([data_train, data_test, model])
gc.collect()


# In[65]:


#dataset #2
cols_svd = ['MGR_ID','RESOURCE_MGR_ID','RESOURCE_ROLE_CODE',
 'RESOURCE_ROLE_DEPTNAME','RESOURCE_ROLE_FAMILY_DESC','RESOURCE_ROLE_FAMILY',
 'RESOURCE_ROLE_ROLLUP_1','RESOURCE','ROLE_CODE',
 'ROLE_DEPTNAME_ROLE_CODE','ROLE_DEPTNAME_ROLE_FAMILY','ROLE_FAMILY_DESC_ROLE_CODE',
 'ROLE_FAMILY_DESC_ROLE_FAMILY','ROLE_FAMILY_DESC','ROLE_ROLLUP_1_ROLE_DEPTNAME',
 'ROLE_ROLLUP_1_ROLE_FAMILY','ROLE_ROLLUP_1_ROLE_ROLLUP_2','ROLE_ROLLUP_2_ROLE_FAMILY_DESC',
 'ROLE_ROLLUP_2_ROLE_FAMILY','ROLE_ROLLUP_2']

cols_rnd = ['MGR_ID_ROLE_CODE','MGR_ID_ROLE_DEPTNAME','MGR_ID_ROLE_FAMILY_DESC',
 'MGR_ID_ROLE_ROLLUP_1','MGR_ID','RESOURCE_ROLE_DEPTNAME',
 'RESOURCE_ROLE_FAMILY','RESOURCE_ROLE_ROLLUP_1','ROLE_CODE',
 'ROLE_DEPTNAME_ROLE_FAMILY_DESC','ROLE_FAMILY_DESC_ROLE_CODE',
 'ROLE_FAMILY_DESC_ROLE_FAMILY','ROLE_FAMILY_ROLE_CODE',
 'ROLE_ROLLUP_1_ROLE_CODE','ROLE_ROLLUP_1_ROLE_FAMILY_DESC',
 'ROLE_ROLLUP_1_ROLE_ROLLUP_2']

cols_freq = ['MGR_ID_ROLE_CODE','MGR_ID_ROLE_DEPTNAME','MGR_ID_ROLE_ROLLUP_1',
 'MGR_ID_ROLE_ROLLUP_2','MGR_ID','RESOURCE_MGR_ID',
 'RESOURCE_ROLE_DEPTNAME','RESOURCE_ROLE_FAMILY_DESC','RESOURCE_ROLE_ROLLUP_2',
 'RESOURCE','ROLE_DEPTNAME_ROLE_FAMILY_DESC','ROLE_DEPTNAME',
 'ROLE_FAMILY_DESC','ROLE_FAMILY','ROLE_ROLLUP_1_ROLE_FAMILY_DESC',
 'ROLE_ROLLUP_1_ROLE_FAMILY','ROLE_ROLLUP_1_ROLE_ROLLUP_2',
 'ROLE_ROLLUP_1','ROLE_ROLLUP_2_ROLE_CODE','ROLE_ROLLUP_2']

cols_te = ['MGR_ID','RESOURCE_MGR_ID','RESOURCE_ROLE_CODE',
 'RESOURCE_ROLE_DEPTNAME','RESOURCE_ROLE_ROLLUP_2','RESOURCE',
 'ROLE_CODE','ROLE_DEPTNAME_ROLE_FAMILY_DESC','ROLE_DEPTNAME_ROLE_FAMILY',
 'ROLE_FAMILY_DESC_ROLE_CODE','ROLE_FAMILY_DESC','ROLE_FAMILY_ROLE_CODE',
 'ROLE_ROLLUP_1_ROLE_FAMILY','ROLE_ROLLUP_2_ROLE_FAMILY','ROLE_ROLLUP_2']

data_svd = transform_dataset(train[cols_svd], test[cols_svd], get_col_interactions_svd)
data_rnd = transform_dataset(train[cols_rnd], test[cols_rnd], 
                             assign_rnd_integer, {"number_of_times":10})
data_freq = transform_dataset(train[cols_freq], test[cols_freq], get_freq_encoding)


# In[66]:


te = TargetEncodingExpandingMean(columns_names=cols_te)

X_tr = te.fit_transform(train[cols_te], y)
X_val = te.transform(test[cols_te])

te2 = TargetEncodingSmoothing(
    columns_names= cols_te,
    k = 3, f = 1.5,
)

X_tr2 = get_CV_target_encoding(train[cols_te], y, te2, cv = 5)
te2.fit(train[cols_te], y)
X_val2 = te2.transform(test[cols_te])

data_te_tr = pd.concat([X_tr, X_tr2], axis = 1)
data_te_te = pd.concat([X_val, X_val2], axis = 1)


# In[67]:


data_train = pd.concat([x[0] for x in [data_svd, data_rnd, data_freq]], axis = 1)
data_test = pd.concat([x[1] for x in [data_svd, data_rnd, data_freq]], axis = 1)
data_train = pd.concat([data_train, data_te_tr], axis = 1)
data_test = pd.concat([data_test, data_te_te], axis = 1)
print("Dataset shape, Train: {}, Test: {}".format(data_train.shape, data_test.shape))
del([data_svd, data_rnd, data_freq, data_te_tr, data_te_te])
gc.collect()
data_train = data_train.values
data_test = data_test.values
gc.collect()


# In[68]:


validate_model(
    model = get_model(), 
    data = [data_train, y]
)


# In[69]:


model = get_model()
model.fit(data_train, y)
predictions_2 = model.predict_proba(data_test)[:,1]

del([data_train, data_test, model])
gc.collect()


# In[70]:


submission = pd.DataFrame()
submission["Id"] = test["id"]
submission["ACTION"] = (predictions_1 + predictions_2) / 2


# In[71]:


submission.to_csv("submission.csv", index = False)

