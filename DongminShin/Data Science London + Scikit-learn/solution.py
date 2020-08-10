#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_data = pd.read_csv('../input/train.csv',header = None)
train_labels = pd.read_csv('../input/trainLabels.csv',header = None)
test_data =  pd.read_csv('../input/test.csv',header = None)


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_labels.head()


# In[6]:


train_data.shape,test_data.shape,train_labels.shape


# In[7]:


train_data.describe()


# ## **PRE-PROCESSING**

# **Train-Test Split**

# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_data,train_labels, test_size = 0.30, random_state = 101)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# ## **CLASSIFICATION**

# In[9]:


# NAIBE BAYES
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train,y_train.values.ravel()) # values.ravel() : y.train에서 목록이 아닌 값들만 1차원 배열로
predicted= model.predict(x_test)
print('Naive Bayes',accuracy_score(y_test, predicted))

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(x_train,y_train.values.ravel())
predicted= knn_model.predict(x_test)
print('KNN',accuracy_score(y_test, predicted))

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)
rfc_model.fit(x_train,y_train.values.ravel())
predicted = rfc_model.predict(x_test)
print('Random Forest',accuracy_score(y_test,predicted))

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver = 'saga')
lr_model.fit(x_train,y_train.values.ravel())
lr_predicted = lr_model.predict(x_test)
print('Logistic Regression',accuracy_score(y_test, lr_predicted))

#SVM
from sklearn.svm import SVC

svc_model = SVC(gamma = 'auto')
svc_model.fit(x_train,y_train.values.ravel())
svc_predicted = svc_model.predict(x_test)
print('SVM',accuracy_score(y_test, svc_predicted))

#DECISON TREE
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier()
dtree_model.fit(x_train,y_train.values.ravel())
dtree_predicted = dtree_model.predict(x_test)
print('Decision Tree',accuracy_score(y_test, dtree_predicted))

#XGBOOST
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train,y_train.values.ravel())
xgb_predicted = xgb.predict(x_test)
print('XGBoost',accuracy_score(y_test, xgb_predicted))


# **KNN, XGBoost, SVM, Random Forest** gave good accuracy using Feature Scaling.

# ## **Feature Scaling**

# In[10]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

X, y = train_data, np.ravel(train_labels)

std = StandardScaler()
X_std = std.fit_transform(X)
mms = MinMaxScaler()
X_mms = mms.fit_transform(X)
norm = Normalizer()
X_norm = norm.fit_transform(X)


# In[11]:


# Model complexity
neig = np.arange(1, 30)
kfold = 10
val_accuracy = {'std':[], 'mms':[], 'norm':[]}
bestKnn = None
bestAcc = 0.0
bestScaling = None
# Loop over different values of k
for i, k in enumerate(neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    # validation accuracy
    s1 = np.mean(cross_val_score(knn, X_std, y, cv=kfold))
    val_accuracy['std'].append(s1)
    s2 = np.mean(cross_val_score(knn, X_mms, y, cv=kfold))
    val_accuracy['mms'].append(s2)
    s3 = np.mean(cross_val_score(knn, X_norm, y, cv=kfold))
    val_accuracy['norm'].append(s3)
    if s1 > bestAcc:
        bestAcc = s1
        bestKnn = knn
        bestScaling = 'std'
    elif s2 > bestAcc:
        bestAcc = s2
        bestKnn = knn
        bestScaling = 'mms'
    elif s3 > bestAcc:
        bestAcc = s3
        bestKnn = knn
        bestScaling = 'norm'

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, val_accuracy['std'], label = 'CV Accuracy with std')
plt.plot(neig, val_accuracy['mms'], label = 'CV Accuracy with mms')
plt.plot(neig, val_accuracy['norm'], label = 'CV Accuracy with norm')
plt.legend()
plt.title('k value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.show()

print('Best Accuracy with feature scaling:', bestAcc)
print('Best kNN classifier:', bestKnn)
print('Best scaling:', bestScaling)


# In[12]:


# NAIBE BAYES
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
print('Naive Bayes',cross_val_score(nb_model,X_norm, train_labels.values.ravel(), cv=10).mean())

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 5)
print('KNN',cross_val_score(knn_model,X_norm, train_labels.values.ravel(), cv=10).mean())

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)
print('Random Forest',cross_val_score(rfc_model,X_norm, train_labels.values.ravel(), cv=10).mean())

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver = 'saga')
print('Logistic Regression',cross_val_score(lr_model,X_norm, train_labels.values.ravel(), cv=10).mean())

#SVM
from sklearn.svm import SVC

svc_model = SVC(gamma = 'auto')
print('SVM',cross_val_score(svc_model,X_norm, train_labels.values.ravel(), cv=10).mean())

#DECISON TREE
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier()
print('Decision Tree',cross_val_score(dtree_model,X_norm, train_labels.values.ravel(), cv=10).mean())

#XGBOOST
from xgboost import XGBClassifier

xgb = XGBClassifier()
print('XGBoost',cross_val_score(xgb,X_norm, train_labels.values.ravel(), cv=10).mean())


# **KNN, XGBoost, SVM, Random Forest** gave good accuracy using Feature Scaling.

# ## **Principal Component Analysis**

# In[13]:


from sklearn.decomposition import PCA

pca  = PCA(n_components=12)
pca_train_data = pca.fit_transform(train_data)
explained_variance = pca.explained_variance_ratio_ 


# In[14]:


pca_train_data.shape


# In[15]:


# NAIBE BAYES
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
print('Naive Bayes',cross_val_score(nb_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())

#KNN
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors = 5)
print('KNN',cross_val_score(knn_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators = 100,random_state = 99)
print('Random Forest',cross_val_score(rfc_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver = 'saga')
print('Logistic Regression',cross_val_score(lr_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())

#SVM
from sklearn.svm import SVC

svc_model = SVC(gamma = 'auto')
print('SVM',cross_val_score(svc_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())

#DECISON TREE

from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier()
print('Decision Tree',cross_val_score(dtree_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())

#XGBOOST
from xgboost import XGBClassifier

xgb = XGBClassifier()
print('XGBoost',cross_val_score(xgb,pca_train_data, train_labels.values.ravel(), cv=10).mean())


# **KNN**, **Random Forest** and **SVM** gave maximum accuracy using Principal Component Analysis.
# 

# # **Feature Selection**

# correlation

# In[16]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(pd.DataFrame(X).corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[17]:


X_train = pd.DataFrame(X, range(700))

corr_matrix = X_train.corr()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.7)]

to_drop


# ## **Applying Gaussian Mixture and Grid Search to improve the accuracy**

# We select the above three algorithms (KNN, Random Forest and SVM) which  gave maximum accuracy for further analysis

# In[18]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

x_all = np.r_[train_data,test_data]
print('x_all shape :',x_all.shape)

# USING THE GAUSSIAN MIXTURE MODEL 
lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)
        gmm.fit(x_all)
        bic.append(gmm.aic(x_all))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
            
best_gmm.fit(x_all)
gmm_train = best_gmm.predict_proba(train_data)
gmm_test = best_gmm.predict_proba(test_data)


#Random Forest Classifier
rfc = RandomForestClassifier(random_state=99)

#USING GRID SEARCH
n_estimators = [10, 50, 100, 200,400]
max_depth = [3, 10, 20, 40]
param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)

grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 10,scoring='accuracy',n_jobs=-1).fit(gmm_train, train_labels.values.ravel())
rfc_best = grid_search_rfc.best_estimator_
print('Random Forest Best Score',grid_search_rfc.best_score_)
print('Random Forest Best Parmas',grid_search_rfc.best_params_)
print('Random Forest Accuracy',cross_val_score(rfc_best,gmm_train, train_labels.values.ravel(), cv=10).mean())

#KNN 
knn = KNeighborsClassifier()

#USING GRID SEARCH
n_neighbors=[3,5,6,7,8,9,10]
param_grid = dict(n_neighbors=n_neighbors)

grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv = 10, n_jobs=-1,scoring='accuracy').fit(gmm_train,train_labels.values.ravel())
knn_best = grid_search_knn.best_estimator_
print('KNN Best Score', grid_search_knn.best_score_)
print('KNN Best Params',grid_search_knn.best_params_)
print('KNN Accuracy',cross_val_score(knn_best,gmm_train, train_labels.values.ravel(), cv=10).mean())

#SVM
svc = SVC()

#USING GRID SEARCH
parameters = [{'kernel':['linear'],'C':[1,10,100]},
              {'kernel':['rbf'],'C':[1,10,100],'gamma':[0.05,0.0001,0.01,0.001]}]
grid_search_svm = GridSearchCV(estimator=svc, param_grid=parameters, cv = 10, n_jobs=-1,scoring='accuracy').fit(gmm_train, train_labels.values.ravel())
svm_best = grid_search_svm.best_estimator_
print('SVM Best Score',grid_search_svm.best_score_)
print('SVM Best Params',grid_search_svm.best_params_)
print('SVM Accuracy',cross_val_score(svm_best,gmm_train, train_labels.values.ravel(), cv=10).mean())


# In[19]:


rfc_best.fit(gmm_train,train_labels.values.ravel())
pred  = rfc_best.predict(gmm_test)
rfc_best_pred = pd.DataFrame(pred)

rfc_best_pred.index += 1

rfc_best_pred.columns = ['Solution']
rfc_best_pred['Id'] = np.arange(1,rfc_best_pred.shape[0]+1)
rfc_best_pred = rfc_best_pred[['Id', 'Solution']]

rfc_best_pred.to_csv('submission.csv',index=False)

