#!/usr/bin/env python
# coding: utf-8

# In[1]:


input_dir = '../input/'
working_dir = '../working/'


# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
test = pd.read_csv(os.path.join(input_dir, 'test.csv'))

# Set index
train.index = train['Id'].values
test.index = test['Id'].values

print(train.shape)
print(test.shape)


# # **EDA**

# In[4]:


train.info()


# In[5]:


train.head(5)


# The data have 9557 entries, each entry has 143 columns.
# 
# Most of the data are floats and integers, a few objects. Let's take a look at the objects.

# # **Object value**

# In[6]:


train.columns[train.dtypes==object]


# Id, idhogar - no problem, they are just identifications
# 
# 
# dependency - dependency rate
# 
# 
# edjefe, edjefa - years of education of head of household
# 
# 

# ### Clean Data
# 
# 
# 1. dependency 'no' -> 0
# 2. edjefa, edjefe 'no' -> 0,  'yes' -> 1
# 3. meaneduc NaN -> mean escolari of household
# 4. v2a1 NaN -> 0
# 5. v18q1 NaN -> 0
# 6. rez_esc NaN -> 0

# 1. dependency
# 
# dependency 'no' -> 0
# 
# we can just derive the dependency from the SQBdependency.
# 
# * So the "square" of no is 0.
# 
# * So the "square" of yes is 1.

# In[7]:


train['dependency'].unique()


# In[8]:


train['SQBdependency'].unique()


# In[9]:


train['SQBdependency']


# In[10]:


train[(train['dependency']=='no') & (train['SQBdependency']!=0)]


# In[11]:


train[(train['dependency']=='yes') & (train['SQBdependency']!=1)]


# In[12]:


train[(train['dependency']=='no') & (train['SQBdependency']!=1)]


# In[13]:


train['dependency']=np.sqrt(train['SQBdependency'])


# 2. edjefa, edjefe 'no' -> 0, 'yes' -> 1
# 
# Basically:
# 
# * 'edjefe' and 'edjefa' are both 'no' when the head of the household had 0 years of school
# * there's 'edjefe'= 'yes' and 'edjefa'='no' in some cases, all these cases the head of the household had 1 year of school
# * there's 'edjefe'= 'no' and 'edjefa'='yes' in some cases, all these cases the head of the household had 1 year of school
# * most of the time either 'edjefe' or 'edjefa' is a number while the other is a 'no'
# * Let's merge the jefe and jefa education into one, undependent of gender

# In[14]:


train['edjefa'].unique()


# In[15]:


train['edjefa'].unique()


# In[16]:


train['SQBedjefe'].unique()


# In[17]:


train[['edjefe', 'edjefa', 'SQBedjefe']][:20]


# 'SQBedjefe is just the square of 'edjefe', it's 0 if the head of the household is a woman.

# In[18]:


train[['edjefe', 'edjefa', 'SQBedjefe']][train['edjefe']=='yes']


# In[19]:


train[(train['edjefe']=='yes') & (train['edjefa']!='no')]


# escolari = years of schooling
# 
# parentesco1 =1 if household head

# In[20]:


train[(train['edjefa']=='yes') & (train['parentesco1']==1)][['edjefe', 'edjefa', 'parentesco1', 'escolari']]


# In[21]:


train[train['edjefe']=='yes'][['edjefe', 'edjefa','age', 'escolari', 'parentesco1','male', 'female', 'idhogar']]


# In[22]:


train[(train['edjefe']=='no') & (train['edjefa']=='no')][['edjefe', 'edjefa', 'age', 'escolari', 'female', 'male', 'Id', 'parentesco1', 'idhogar']]


# In[23]:


conditions = [
    (train['edjefe']=='no') & (train['edjefa']=='no'), #both no
    (train['edjefe']=='yes') & (train['edjefa']=='no'), # yes and no
    (train['edjefe']=='no') & (train['edjefa']=='yes'), #no and yes 
    (train['edjefe']!='no') & (train['edjefe']!='yes') & (train['edjefa']=='no'), # number and no
    (train['edjefe']=='no') & (train['edjefa']!='no') # no and number
]
choices = [0, 1, 1, train['edjefe'], train['edjefa']]
train['edjefx']=np.select(conditions, choices)
train['edjefx']=train['edjefx'].astype(int)
train[['edjefe', 'edjefa', 'edjefx']][:15]


# # **missing values**

# In[24]:


train.columns[train.isna().sum()!=0]


# Columns with nans:
# 
# * v2a1 - monthly rent
# * v18q1 - number of tablets
# * rez_esc - years behind school
# * meaneduc - mean education for adults
# * SQBmeaned - square of meaned

# 3. meaneduc NaN -> mean escolari of household
# 
# 'meaneduc' and 'SQBmeaned' are related

# In[25]:


train[train['meaneduc'].isnull()]


# In[26]:


train[train['meaneduc'].isnull()][['Id','idhogar','edjefe','edjefa', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'age', 'escolari']]


# So, the 5 rows with Nan for 'meaneduc' is just 3 households, where 18-19 year-olds live. No other people live in these households. Then we can just take the education levels of these kids ('escolari') and put them into 'meaneduc' and 'SQBmeaned'.

# 4. v2a1 NaN -> 0
# 
# Next, let's look at 'v2a1', the monthly rent payment, that also has missing values.

# In[27]:


norent=train[train['v2a1'].isnull()]
print("Owns his house:", norent[norent['tipovivi1']==1]['Id'].count())
print("Owns his house paying installments", norent[norent['tipovivi2']==1]['Id'].count())
print("Rented ", norent[norent['tipovivi3']==1]['Id'].count())
print("Precarious ", norent[norent['tipovivi4']==1]['Id'].count())
print("Other ", norent[norent['tipovivi5']==1]['Id'].count())
print("Total ", 6860)


# The majority in fact owns their houses, only a few have odd situations. We can probably just assume they don't pay rent, and put 0 in these cases.

# 5. v18q1 NaN -> 0
# 
# let's look at 'v18q1', which indicates how many tablets the household owns.

# In[28]:


train['v18q1'].unique()


# 6. rez_esc NaN -> 0
# 
# rez_esc
#  : Years behind in school

# In[29]:


rez_esc_nan=train[train['rez_esc'].isnull()]
rez_esc_nan[(rez_esc_nan['age']<18) & rez_esc_nan['escolari']>0][['age', 'escolari']]


# So all the nans here are either adults or children before school age. We can input 0 again

# In[30]:


def data_cleaning(data):
    data['dependency']=np.sqrt(data['SQBdependency'])
    data['rez_esc']=data['rez_esc'].fillna(0)
    data['v18q1']=data['v18q1'].fillna(0)
    data['v2a1']=data['v2a1'].fillna(0)
    
    conditions = [
    (data['edjefe']=='no') & (data['edjefa']=='no'), #both no
    (data['edjefe']=='yes') & (data['edjefa']=='no'), # yes and no
    (data['edjefe']=='no') & (data['edjefa']=='yes'), #no and yes 
    (data['edjefe']!='no') & (data['edjefe']!='yes') & (data['edjefa']=='no'), # number and no
    (data['edjefe']=='no') & (data['edjefa']!='no') # no and number
    ]
    choices = [0, 1, 1, data['edjefe'], data['edjefa']]
    data['edjefx']=np.select(conditions, choices)
    data['edjefx']=data['edjefx'].astype(int)
    data.drop(['edjefe', 'edjefa'], axis=1, inplace=True)
    
    meaneduc_nan=data[data['meaneduc'].isnull()][['Id','idhogar','escolari']]
    me=meaneduc_nan.groupby('idhogar')['escolari'].mean().reset_index()
    for row in meaneduc_nan.iterrows():
        idx=row[0]
        idhogar=row[1]['idhogar']
        m=me[me['idhogar']==idhogar]['escolari'].tolist()[0]
        data.at[idx, 'meaneduc']=m
        data.at[idx, 'SQBmeaned']=m*m
        
    return data


# In[31]:


train = data_cleaning(train)
test = data_cleaning(test)


# ### Extract heads of household

# In[32]:


train = train.query('parentesco1==1')
train = train.drop('parentesco1', axis=1)
test = test.drop('parentesco1', axis=1)
print(train.shape)


# ## Convert one-hot variables into numeric
# * 'epared', 'etecho', 'eviv' and 'instlevel' can be converted into numeric
# *  like (bad, regular, good) -> (0 ,1, 2)

# In[33]:


def get_numeric(data, status_name):
    # make a list of column names containing 'sataus_name'
    status_cols = [s for s in data.columns.tolist() if status_name in s]
    print('status column names')
    print(status_cols)
    # make a DataFrame with only status_cols
    status_df = data[status_cols]
    # change its column name like ['epared1', 'epared2', 'epared3'] -> [0, 1, 2]
    status_df.columns = list(range(status_df.shape[1]))
    # get the column name which has the biggest value in every row
    
    # this is pandas.Series
    status_numeric = status_df.idxmax(1)
    # set Series name
    status_numeric.name = status_name
    # add status_numeric as a new column
    data = pd.concat([data, status_numeric], axis=1)
    return data


# In[34]:


status_name_list = ['epared', 'etecho', 'eviv', 'instlevel']
for status_name in status_name_list:
    train = get_numeric(train, status_name)
    test = get_numeric(test, status_name)


# ## Delete needless columns
# ### redundant columns
# * r4t3, tamviv, tamhog, hhsize ... almost the same as hogar_total
# * v14a ... almost the same as saniatrio1
# * v18q, mobilephone ... can be generated by v18q1, qmobilephone
# * SQBxxx, agesq ... squared values
# * parentescoxxx ... only heads of household are in dataset now
# 
# ### extra columns
# (One-hot variables should be linearly independent. For example, female (or male) column is needless, because whether the sample is female or not can be explained only with male (or female) column.)
# * paredother, pisoother, abastaguano, energcocinar1, techootro, sanitario6, elimbasu6, estadocivil7, parentesco12, tipovivi5, lugar1, area1, female
# 
# ### obsolete columns
# * epared1~3, etecho1~3, eviv1~3, instlevel1~9 ... we don't use these columns anymore.
# 

# In[35]:


needless_cols = ['r4t3', 'tamhog', 'tamviv', 'hhsize', 'v18q', 'v14a', 'agesq',
                 'mobilephone', 'paredother', 'pisoother', 'abastaguano',
                 'energcocinar1', 'techootro', 'sanitario6', 'elimbasu6',
                 'estadocivil7', 'parentesco12', 'tipovivi5',
                 'lugar1', 'area1', 'female', 'epared1', 'epared2',
                 'epared3', 'etecho1', 'etecho2', 'etecho3',
                 'eviv1', 'eviv2', 'eviv3', 'instlevel1', 'instlevel2',
                 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6',
                 'instlevel7', 'instlevel8', 'instlevel9']
SQB_cols = [s for s in train.columns.tolist() if 'SQB' in s]
parentesco_cols = [s for s in train.columns.tolist() if 'parentesco' in s]

needless_cols.extend(SQB_cols)
needless_cols.extend(parentesco_cols)

train = train.drop(needless_cols, axis=1)
test = test.drop(needless_cols, axis=1)


# In[36]:


ori_train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
ori_train_X = ori_train.drop(['Id', 'Target', 'idhogar'], axis=1)

train_X = train.drop(['Id', 'Target', 'idhogar'], axis=1)

print('feature columns \n {} -> {}'.format(ori_train_X.shape[1], train_X.shape[1]))


# ## Simple LightGBM

# In[37]:


# Split data
train_Id = train['Id'] # individual ID
train_idhogar = train['idhogar'] # household ID
train_y = train['Target'] # Target value
train_X = train.drop(['Id', 'Target', 'idhogar'], axis=1) # features

test_Id = test['Id'] # individual ID
test_idhogar = test['idhogar'] # household ID
test_X = test.drop(['Id', 'idhogar'], axis=1) # features

# Union train and test
all_Id = pd.concat([train_Id, test_Id], axis=0, sort=False)
all_idhogar = pd.concat([train_idhogar, test_idhogar], axis=0, sort=False)
all_X = pd.concat([train_X, test_X], axis=0, sort=False)


# In[38]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
import lightgbm as lgb

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.1, random_state=0)

F1_scorer = make_scorer(f1_score, greater_is_better=True, average='macro')

# gbm_param = {
#     'num_leaves':[210]
#     ,'min_data_in_leaf':[9]
#     ,'max_depth':[14]
# }
# gbm = GridSearchCV(
#     lgb.LGBMClassifier(objective='multiclassova', class_weight='balanced', seed=0)
#     , gbm_param
#     , scoring=F1_scorer
# )


# params = {'num_leaves': 13, 'min_data_in_leaf': 23, 'max_depth': 11, 'learning_rate': 0.09, 'feature_fraction': 0.74}
gbm = lgb.LGBMClassifier(boosting_type='dart', objective='multiclassova', class_weight='balanced', random_state=0)
# gbm.set_params(**params)

gbm.fit(X_train, y_train)
# gbm.best_params_


# In[39]:


import pickle
with open(os.path.join(working_dir, '20180801_lgbm.pickle'), mode='wb') as f:
    pickle.dump(gbm, f)


# In[40]:


y_test_pred = gbm.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='macro')
print("confusion matrix: \n", cm)
print("macro F1 score: \n", f1)


# In[41]:


pred = gbm.predict(test_X)
pred = pd.Series(data=pred, index=test_Id.values, name='Target')
pred = pd.concat([test_Id, pred], axis=1, join_axes=[test_Id.index])
pred.to_csv('submission.csv', index=False)

