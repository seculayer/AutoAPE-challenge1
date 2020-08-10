#!/usr/bin/env python
# coding: utf-8

# ## Walmart Recruiting: Trip Type Classification

# ### Objective: 
# 
# - Trip Type Classification of each customers based on thier shopping data
# 

# ### Data Description: 
# 
# - train : 647054 rows, 7 columns
# - test : 653646 rows, 6 columns
# 
# | Index | Feature               | Feature Description                                  | Unique Value |
# |-------|-----------------------|----------------------------------------------|--------|
# | 1     | TripType              | A categorical id representing the type of shopping trip the customer made.                                       | 38     |
# | 2     | VisitNumber           | An id corresponding to a single trip by a single customer                              | 95674  |
# | 3     | Weekday               | The weekday of the trip                    | 7      |
# | 4     | Upc                   | The UPC number of the product purchased                  | 97715  |
# | 5     | ScanCount             | The number of the given item that was purchased. A negative value indicates a product return          | 39     |
# | 6     | DepartmentDescription | A high-level description of the item's department                                | 69     |
# | 7     | FinelineNumber        | A more refined category for each of the products, created by Walmart | 5196   |

# ### UPC code description
# <img src="https://github.com/novdov/dss7_SWYA_walmart/blob/master/main/data/upc.png?raw=true", width="550">

# ### Evaluation : Multi-class log loss (Cross Entropy)
#  $$-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij})$$
# 
# - $N$ : the number of visits in the test set
# - $M$ : the number of trip types
# - $\log$ : natural logarithm
# - $y_{ij}$ : 1 if observation i is of class j and 0 otherwise
# - $p_{ij}$ : the predicted probability that observation i belongs to class j

# ## Contents
# 
# - **[1. EDA & Preprocessing](#1.-EDA-&-Preprocessing)**  
# <br>
# 
# - **[2. Feature Engineering](#2.-Feature-Engineering)**
#     - UPC decoding
#     - ScanCount seperation
#     - Feature encoding
#     - Dummy variables
#     - Identifing the most frequently purchased items per VisitNumber
# 
# <br>
# - **[3. Modeling](#3.-Modeling)**
#     - XGBoost
# 

# # 1. EDA & Preprocessing

# In[46]:


import numpy as np
import pandas as pd
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'png', 'retina'}")


# In[47]:


train = pd.read_csv("../input/walmart/train.csv")
test = pd.read_csv("../input/walmart/test.csv")


# In[48]:


print(train.shape)
train.tail()


# In[49]:


train.info()


# In[50]:


sample = pd.read_csv("../input/walmart/sample_submission.csv")
sample.head(5)


# ### Missing Values

# In[51]:


plt.figure(figsize=(7, 5))
train.isnull().sum().plot(kind='bar')
plt.xticks(rotation=45, ha='right')
plt.show()


# - Upc, DepartmentDescription, FinelineNumber에 Missing Values 존재
# - Upc와 FinelineNumber의 Missing Values는 동시에 나타남
# - Missing Value 제거 시 사라지는 VisitNumber 존재
# - Feature Engineering에서 VisitNumber에 대한 테이블로 변경
# - 최종 모델링 시 Nan은 0으로 처리

# ### Encode Weekday (Labeling)

# In[52]:


wd = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
      "Friday": 4, "Saturday": 5, "Sunday": 6}
train["Weekday"] = train["Weekday"].apply(lambda x: wd[x])
test["Weekday"] = test["Weekday"].apply(lambda x: wd[x])


# In[53]:


plt.figure(figsize=(14, 3))

sns.heatmap(data=pd.crosstab(train["Weekday"],
                             train["TripType"], 
                             values=train["VisitNumber"],
                             aggfunc='count', 
                             normalize="columns"), cmap="ocean_r")
plt.yticks(range(0,7), list(wd.keys()), rotation="horizontal")
plt.title("Distribution of Triptype on Weekday", fontsize=13)
plt.show()


# - TripType 14는 월, 금/토에 많이 나타남
# - Feature에 반영 (One-Hot Encoding)

# ### Uneven Distribution of TripType

# In[54]:


plt.figure(figsize=(12, 5))

sns.set_style('whitegrid')
np.sort(train.TripType.unique())
train_triptypes = train.drop_duplicates("VisitNumber")
a = train_triptypes["TripType"]
a = a.value_counts()
a.plot(kind='bar', color="lightseagreen")
plt.title("Number of TripType Occurence by VisitNumber", fontsize=13)
plt.xlabel("TripType")

plt.show()


# ### Most Frequent & Least Frequent TripType

# In[56]:


fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
type_8 = train[train.TripType == 8]
type_8_items = type_8[["TripType","DepartmentDescription"]]
type_8_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                        title="TripType 8", color="seagreen")
ax1 = fig.add_subplot(122)
type_14 = train[train.TripType == 14]
type_14_items = type_14[["TripType","DepartmentDescription"]]
type_14_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                        title="TripType 14", color="lightcoral")

fig.suptitle("Distrubution of DepartmentDescription each TripType", fontsize=13)
plt.xticks(fontsize=10)
plt.show()


# In[57]:


trip_desc = pd.crosstab(train["TripType"], 
                        train["DepartmentDescription"], 
                        values=train["ScanCount"], 
                        aggfunc="count", 
                        normalize="index")

plt.figure(figsize=(18, 9))
sns.heatmap(trip_desc, linecolor="lightgrey", linewidths=0.02, cmap="RdPu", alpha=.8)
plt.title("Distrubution of DepartmentDescription on TripTypes", fontsize=13)
plt.show()


# - 비슷한 Department에서 물건 구매시 TripType 분류에 영향을 미치는 것으로 보임
# - 하나의 Department에서만 구매한 것이 아닌 여러 Department에서 구매함
# - 복수의 Department 방문 여부를 Feature에 반영 (CategoryCounts)

# In[58]:


train_plot = train[["TripType", "ScanCount"]]
plot_grouped = train_plot.groupby("TripType", as_index=False).sum()

f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 5))
sns.barplot(x="TripType", y="ScanCount", data=plot_grouped, ax=ax)
sns.barplot(x="TripType", y="ScanCount", data=plot_grouped, ax=ax2)

ax.set_ylim(100000, 210000)
ax2.set_ylim(-2000, 2000)
ax.xaxis.label.set_visible(False)
ax.yaxis.label.set_visible(False)
ax2.yaxis.label.set_visible(False)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off') 
ax2.xaxis.tick_bottom()

d = .005

kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  

kwargs.update(transform=ax2.transAxes) 
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) 

f.text(0, 0.5, 'ScanCount', va='center', rotation='vertical', fontsize=11)
f.text(0.5, 1, 'ScanCount', va='center', rotation='horizontal', fontsize=13)
plt.tight_layout()
plt.show()


# - TripType 999는 반품을 위한 방문이라고 생각됨
# - Return 여부를 Feature에 반영 (Return)
# - ScanCount를 실제 판매수량과 반품으로 나누어서 Feature에 반영 (Pos_Sum, Neg_Sum)

# # 2. Feature Engineering

# ### Decode company code from UPC

# In[59]:


def float_to_str(obj):
    """
    Convert Upc code from float to string
    Use this function by applying lambda
    :param obj: "Upc" column of DataFrame
    :return: string converted Upc removing dot.
    """
    while obj != "nan":
        obj = str(obj).split(".")[0]
        return obj


# In[60]:


def company(x):
    """
    Return company code from given Upc code.
    :param x: "Upc" column of DataFrame
    :return: company code
    """
    try:
        p = x[:6]
        if p == "000000":
            return x[-5]
        return p
    except:
        return -9999


# In[61]:


train["Upc"] = train.Upc.apply(float_to_str)
test["Upc"] = test.Upc.apply(float_to_str)

train["company"] = train.Upc.apply(company) 
test["company"] = test.Upc.apply(company)


# ### Return Indicator

# In[62]:


train.loc[train["ScanCount"] < 0, "Return"] = 1
train.loc[train["Return"] != 1, "Return"] = 0

test.loc[test["ScanCount"] < 0, "Return"] = 1
test.loc[test["Return"] != 1, "Return"] = 0


# ### Positive ScanCount (The number of sold)

# In[63]:


train["Pos_Sum"] = train["ScanCount"]
test["Pos_Sum"] = test["ScanCount"]

train.loc[train["Pos_Sum"] < 0, "Pos_Sum"] = 0
test.loc[test["Pos_Sum"] < 0, "Pos_Sum"] = 0


# ### Negative ScanCount (The number of returns)

# In[64]:


train["Neg_Sum"] = train["ScanCount"]
test["Neg_Sum"] = test["ScanCount"]

train.loc[train["Neg_Sum"] > 0, "Neg_Sum"] = 0
test.loc[test["Neg_Sum"] > 0, "Neg_Sum"] = 0


# In[65]:


print(train.shape)
train.tail()


# ### FinelineNumber most frequently appear

# In[66]:


def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    ls = [x_i for x_i, count in counts.items() if count == max_count]
    return ls[0]


# In[67]:


train_fineline = train[["VisitNumber", "FinelineNumber"]]
train_fineline = train_fineline.groupby("VisitNumber", as_index=False).agg(mode)
train_fineline.rename(columns={"FinelineNumber": "MF_FinelineNumber"}, inplace=True)

test_fineline = test[["VisitNumber", "FinelineNumber"]]
test_fineline = test_fineline.groupby("VisitNumber", as_index=False).agg(mode)
test_fineline.rename(columns={"FinelineNumber": "MF_FinelineNumber"}, inplace=True)


# In[68]:


train_fineline.head(5)
test_fineline.head(5)


# ### The number of UPC and  FinelineNumber for each VisitNumber

# In[69]:


train_upc_fine = train[["VisitNumber", "Upc", "FinelineNumber"]]
test_upc_fine = test[["VisitNumber", "Upc", "FinelineNumber"]]

train_upc_fine_group = train_upc_fine.groupby("VisitNumber", as_index=False).count()
train_upc_fine_group.rename(columns={"Upc": "N_Upc", 
                                     "FinelineNumber": "N_FinelineNumber"}, inplace=True)
test_upc_fine_group = test_upc_fine.groupby("VisitNumber", as_index=False).count()
test_upc_fine_group.rename(columns={"Upc": "N_Upc", 
                                    "FinelineNumber": "N_FinelineNumber"}, inplace=True)


# In[70]:


train_upc_fine_group.head(5)
test_upc_fine_group.head(5)


# In[71]:


train.drop(["Upc", "FinelineNumber"], axis=1, inplace=True)
test.drop(["Upc", "FinelineNumber"], axis=1, inplace=True)


# ### One-Hot Encoding of DepartmentDescription

# In[72]:


train_dd = pd.get_dummies(train["DepartmentDescription"])
test_dd = pd.get_dummies(test["DepartmentDescription"])

train_dd = pd.concat([train[["VisitNumber"]], train_dd], axis=1)
test_dd = pd.concat([test[["VisitNumber"]], test_dd], axis=1)

train_dd = train_dd.groupby("VisitNumber", as_index=False).sum()
test_dd = test_dd.groupby("VisitNumber", as_index=False).sum()


# In[73]:


train_dd.tail()


# In[74]:


train_company = train[["VisitNumber", "company"]]
test_company = test[["VisitNumber", "company"]]

train_company = train_company.groupby("VisitNumber", as_index=False).agg(mode)
test_company = test_company.groupby("VisitNumber", as_index=False).agg(mode)


# In[75]:


train_company.head(5)
test_company.head(5)


# ### Pivot data by VisitNumber

# In[76]:


train_by_sum = train[["VisitNumber", "ScanCount", "Pos_Sum", "Neg_Sum"]]
test_by_sum = test[["VisitNumber", "ScanCount", "Pos_Sum", "Neg_Sum"]]

train_by_sum = train_by_sum.groupby("VisitNumber", as_index=False).sum()
test_by_sum = test_by_sum.groupby("VisitNumber", as_index=False).sum()

train_by_max = train[["TripType", "VisitNumber", "Weekday", "Return"]]
test_by_max = test[["VisitNumber", "Weekday", "Return"]]

train_by_max = train_by_max.groupby("VisitNumber", as_index=False).max()
test_by_max = test_by_max.groupby("VisitNumber", as_index=False).max()


# In[77]:


train = train_by_sum.merge(train_by_max, on=["VisitNumber"])
train = train.merge(train_dd, on=["VisitNumber"])
train = train.merge(train_company, on=["VisitNumber"])
train = train.merge(train_fineline, on=["VisitNumber"])
train = train.merge(train_upc_fine_group, on=["VisitNumber"])

test = test_by_sum.merge(test_by_max, on=["VisitNumber"])
test = test.merge(test_dd, on=["VisitNumber"])
test = test.merge(test_company, on=["VisitNumber"])
test = test.merge(test_fineline, on=["VisitNumber"])
test = test.merge(test_upc_fine_group, on=["VisitNumber"])


# In[78]:


train.head(5)
test.head(5)


# - Remove DepartmentDescription not shown in test data
# - Segregate features & target

# In[79]:


y = train["TripType"]
train = train.drop(["TripType", "HEALTH AND BEAUTY AIDS"], axis=1)


# ### The counts of DepartmentDescription for each VistNumber

# In[80]:


def category_counts(data):
    """
    Count total number of unique DepartmentDescription made on each trip.
    """
    counts = []
    for array in np.asarray(data.loc[:, "1-HR PHOTO":"WIRELESS"]):
        count = 0
        for item in array:
            if item > 0:
                count += 1
        counts.append(count)
    cat_counts = pd.DataFrame(counts)
    cat_counts = cat_counts.rename(columns={0: "CategoryCount"})
    cat_counts = cat_counts.set_index(data.index)

    data.insert(6, "CategoryCounts", cat_counts)

    return data


# In[81]:


# from walmart_utils import category_counts
get_ipython().run_line_magic('time', 'train = category_counts(train)')
get_ipython().run_line_magic('time', 'test = category_counts(test)')


# In[82]:


train.head(5)
test.head(5)


# ### Ratio of number of UPC and FinelineNumber

# In[83]:


train["Upc_FLN"] = train["N_Upc"] / train["N_FinelineNumber"]
test["Upc_FLN"] = test["N_Upc"] / test["N_FinelineNumber"]


# In[84]:


train.head(5)
test.head(5)


# ### One-Hot Encoding of Weekday and Return

# In[85]:


train = pd.get_dummies(train, columns=["Weekday", "Return"])
test = pd.get_dummies(test, columns=["Weekday", "Return"])


# In[86]:


train.head(5)


# In[87]:


vn = test[["VisitNumber"]]
train.drop("VisitNumber", axis=1, inplace=True)
test.drop("VisitNumber", axis=1, inplace=True)


# In[88]:


train.head(5)


# ### Replace Null Value, Inf Value with 0

# In[89]:


train.replace(np.inf, 0, inplace=True)
train.fillna(value=0, inplace=True)

test.replace(np.inf, 0, inplace=True)
test.fillna(value=0, inplace=True)


# In[90]:


train.head(5)


# ### Final Data
# - 85 features

# ### Final Features (Total 85)
# 
# - ScanCount
# - Return (dummies)
# - Pos_Sum
# - Neg_Sum
# - CategoryCounts
# - DepartmentDescription (dummies)
# - Company code
# - MF_Fineline (Most Frequent FinelineNumbers)
# - MF_DepartmentDescription (Most Frequent DepartmentDescription)
# - N_UPC (The number of UPC)
# - N_FinelineNumber (The number of FinelineNumber)
# - UPC_FLN (N_UPC / N_FinelineNumber)
# - Weekday (dummies)

# # 3. Modeling
# 
# - XGBoost

# In[91]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[92]:


label_enc = LabelEncoder().fit(y)
y_labeled = label_enc.transform(y)


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(
    train, y_labeled, random_state=0)


# In[94]:


dtrain = xgb.DMatrix(X_train.values, label=y_train)
dtest = xgb.DMatrix(X_test.values, label=y_test)


# In[95]:


num_boost_round = 300
params = {'objective': 'multi:softprob', 
          'eval_metric': 'mlogloss',
          'num_class':38, 
          'max_delta_step': 3, 
          'eta': 0.2}

evals = [(dtrain, 'train'), (dtest, 'eval')]

bst = xgb.train(params=params,  
                dtrain=dtrain, 
                num_boost_round=num_boost_round, 
                evals=evals,
                early_stopping_rounds=10,)


# In[96]:


from sklearn.metrics import log_loss
my_test = xgb.DMatrix(X_test.values)
test_predictions = bst.predict(my_test)
print("log loss :", log_loss(y_test, test_predictions).round(5))


# In[97]:


import operator
importance_dict = bst.get_score()
sorted_dict = sorted(importance_dict.items(), key=operator.itemgetter(1))
sorted_dict = sorted_dict[::-1]

indices = []
for i in range(len(sorted_dict)-1):
    indices.append(sorted_dict[i][0])
    
indices = [int(idx[1:]) for idx in indices]

importance_features = []
for idx in indices:
    importance_features.append(train.columns[idx])

importance = list(bst.get_score().values())
importance = sorted(importance)
importance = importance[::-1]

pairs = list(zip(importance, importance_features))
labels = [label[1] for label in pairs]

plt.figure(figsize=(12, 8))
plt.barh(range(20), importance[:20][::-1], align="center")
plt.yticks(np.arange(20), labels[:20][::-1])
plt.xlabel("Feature importance")
plt.ylim(-1, 20)
plt.show()


# ##### Accuracy score

# In[98]:


yprob = test_predictions.reshape(y_test.shape[0], 38)
ylabel = np.argmax(yprob, axis=1)
accuracy = 100 *(1 - (sum(int(ylabel[i]) != y_test[i]
                         for i in range(len(y_test))) / float(len(y_test))))
print("TripType Accuracy = %.2f%%" % accuracy)


# ### Submission

# In[99]:


classes = np.array(list(set(label_enc.inverse_transform(y_labeled))))

dmtest = xgb.DMatrix(test.values) 
pred_proba = bst.predict(dmtest)

proba_df = pd.DataFrame(pred_proba, columns=classes)
proba_df.columns = proba_df.columns.map(lambda x: "TripType_" + str(x))
sub_df = pd.concat([vn, proba_df], axis=1)

sub_df.to_csv("submission.csv", index=False)

