#!/usr/bin/env python
# coding: utf-8

# # QDA + Pseudo Labeling + Gaussian Mixture = LB 0.975
# The dataset for Kaggle competition "Instant Gratification" appears to be 512 datasets concatenated where each sub dataset is believed to be created by Sklearn's `make_classification`. EDA suggests the following parameters:
#   
#     X, y = make_classification(n_samples=1024, n_features=255, n_informative=33+x,
#                 n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=3,
#                 weights=None, flip_y=0.05, class_sep=1.0, hypercube=True, shift=0.0, 
#                 scale=1.0, shuffle=True, random_state=None) # where 0<=x<=14
#                 
# The important parameters to note are `n_clusters_per_class=3` and `n_informative=33+x`. This means that the data resides in `33+x` dimensional space within 6 hyper-ellipsoids. Each hyper-ellipsoid is a multivariate Gaussian distribution therefore the best classifiers to use are QDA, Pseudo Labeling, and Gaussian Mixture. (See appendix for EDA showing 3 clusters per class).
# 
# ![image](http://playagricola.com/Kaggle/six62019.jpg)
# 

# # Load Libraries and Data

# In[1]:


# IMPORT LIBRARIES
import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


# In[2]:


# LOAD TRAIN AND TEST
df_train = pd.read_csv('../input/instant-gratification/train.csv')
df_test = pd.read_csv('../input/instant-gratification/test.csv')
sample = pd.read_csv('../input/instant-gratification/sample_submission.csv')


# In[3]:


df_train.info()


# In[4]:


df_test.info()


# In[5]:


df_train.head()


# In[6]:


df_test.head()


# In[7]:


print("Train shape: {}, Test shape: {}".format(df_train.shape, df_test.shape))


# Interactions
# 
# Below shows the interactions between wheezy-copper-turtle-magic and the other variables. Each variable by itself cannot predict target well, but when wheezy-copper-turtle-magic equals a specific value then other variables can predict target well. For example, when wheezy-copper-turtle-magic = 0 then zippy-harlequin-otter-grandmaster is positively correlated with target. And when wheezy-copper-turtle-magic = 0 then hasty-puce-fowl-fepid is negatively correlated with target.

# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.figure(figsize=(15,5))

# PLOT ALL ZIPPY
plt.subplot(1,2,1)
sns.distplot(df_train[ (df_train['target']==0) ]['zippy-harlequin-otter-grandmaster'], label = 't=0')
sns.distplot(df_train[ (df_train['target']==1) ]['zippy-harlequin-otter-grandmaster'], label = 't=1')
plt.title("Without interaction, zippy has no correlation \n (showing all rows)")
plt.xlim((-5,5))
plt.legend()

# PLOT ZIPPY WHERE WHEEZY-MAGIC=0
plt.subplot(1,2,2)
sns.distplot(df_train[ (df_train['wheezy-copper-turtle-magic']==0) & (df_train['target']==0) ]
             ['zippy-harlequin-otter-grandmaster'], label = 't=0')
sns.distplot(df_train[ (df_train['wheezy-copper-turtle-magic']==0) & (df_train['target']==1) ]
             ['zippy-harlequin-otter-grandmaster'], label = 't=1')
plt.title("With interaction, zippy has postive correlation \n (only showing rows where wheezy-copper-turtle-magic=0)")
plt.legend()

plt.show()


# In[9]:


plt.figure(figsize=(15,5))

# PLOT ALL HASTY
plt.subplot(1,2,1)
sns.distplot(df_train[ (df_train['target']==0) ]['hasty-puce-fowl-fepid'], label = 't=0')
sns.distplot(df_train[ (df_train['target']==1) ]['hasty-puce-fowl-fepid'], label = 't=1')
plt.title("Without interaction, hasty has no correlation \n (showing all rows)")
plt.xlim((-5,5))
plt.legend()

# PLOT HASTY WHERE WHEEZY-MAGIC=0
plt.subplot(1,2,2)
sns.distplot(df_train[ (df_train['wheezy-copper-turtle-magic']==0) & (df_train['target']==0) ]
             ['hasty-puce-fowl-fepid'], label = 't=0')
sns.distplot(df_train[ (df_train['wheezy-copper-turtle-magic']==0) & (df_train['target']==1) ]
             ['hasty-puce-fowl-fepid'], label = 't=1')
plt.title("With interaction, hasty has negative correlation \n (only showing rows where wheezy-copper-turtle-magic=0)")
plt.legend()

plt.show()


# In[14]:


df_train = pd.read_csv('../input/instant-gratification/train.csv')
df_test = pd.read_csv('../input/instant-gratification/test.csv')


# Structure of Entire Dataset
# 
# The dataset for Kaggle's "Instant Gratification" competition appears to be 512 partial datasets combined as confirmed by experiments and EDA. Consequently each variable (column) appears to be the sum of 512 Gaussians. Below is a sketch of the dataset sorted by the column wheezy-copper-turtle-magic (using 1-based indexing).
# 
# ![image.png](attachment:image.png)
# 

# In the dataset drawing above, column one is the summation of Gaussians DATA-1-1, DATA-2-1, and DATA-3-1. If you plot these histograms separately, you see that they are Gaussians.
# 
# ![image.png](attachment:image.png)
# 
# **Structure of Partial Datasets**
# 
# **There are 512 unique values of wheezy-copper-turtle-magic and there are 262144 rows of data. For each value of magic there are approximately 512 = 262144/512 associated rows that form a partial dataset.**
# 
# 
# **These partial datasets each have (approx) 512 rows and 255 features. We can model each partial dataset individually** and this can achieve high overall accuracy of at least LB 0.928 as shown here. (Predictions on test.csv are made using the appropriate model). The features for model one are DATA-1-1, DATA-1-2, DATA-1-3, etc. Each feature has a Gaussian distribution. When building a model with only 512 observations and 255 features, it is important to reduce the number of features to prevent overfitting.

# Determine structure of private dataset
# 
# As described here, **both the train and public test dataset appear to be 512 datasets combined. Each partial dataset has 256 variables. Therefore there are 131072 = 512 * 216 blocks of data. Each block of data has either standard deviation approx 1.0 or approx 3.75. The blocks with standard deviation 1.0 seem to be useless while the blocks with standard deviation 3.75 seem to be useful.** Does the private test dataset have this same structure?
# 
# (There are 512 partial datasets within the full dataset. **And each partial dataset has a different set of approximately 40 important features as identified by different standard deviations.**) We have observed that both the training dataset and public test dataset have this structure. Does the private dataset also have this structure? In this kernel, we probe the private dataset and confirm that it has the same structure.

# In[15]:


print("Train shape: {}, Test shape: {}".format(df_train.shape, df_test.shape))


# In[16]:


df_train.describe()


# In[17]:


df_test.head()


# In[18]:


sample.head()


# # Identify Useful Features
# the data appears to be 131072 = 512 * 256 blocks of data where some are useful and some are useless.

# In[19]:


# IDENTIFY USEFUL FEATURES PER MAGIC SUB-DATASET
useful = np.zeros((256,512))
for i in range(512):
    partial = df_train[ df_train['wheezy-copper-turtle-magic']==i ]
    useful[:,i] = np.std(partial.iloc[:,1:-1], axis=0)
useful = useful > 1.5


# In[20]:


import matplotlib.pyplot as plt

# PLOT BOOLEANS OF USEFUL BLOCKS
plt.figure(figsize=(10,20))
plt.matshow(useful.transpose(),fignum=1)
plt.title('The useful datablocks of dataset', fontsize=24)
plt.xlabel('256 Variable columns', fontsize=16)
plt.ylabel('512 Partial datasets', fontsize=16)
plt.show()


# In[21]:


useful = np.sum( useful, axis=0 )


# What is Pseudo Labeling?
# 
# Pseudo labeling is the process of adding confident predicted test data to your training data. Pseudo labeling is a 5 step process. (1) Build a model using training data. (2) Predict labels for an unseen test dataset. (3) Add confident predicted test observations to our training data (4) Build a new model using combined data. And (5) use your new model to predict the test data and submit to Kaggle. Here is a pictorial explanation using sythetic 2D data.

# Why does Pseudo Labeling work?
# 
# Pseudo labeling helps all types of models because all models can be visualized as finding shapes of target=1 and target=0 in p-dimensional space. See here for examples. More points allow for better estimation of shapes.

# # Model and Predict
# 1. First we use QDA plus Pseudo Labeling 
# 2. Next we will use these predictions (pseudo labels) to find the 6 ellipses. We separately find the 3 ellipses of the target=1 data and 3 ellipses of the target=0 data using Sklearn GaussianMixture. 
# 3. Then we label each point with 0, 1, 2, 3, 4, 5 representing which ellipse it belongs to. 
# 4. Finally we train QDA on these 6 ellipses and use QDA to make our final predictions with `Pr(target=1) = Pr(in ellipse 3) + Pr(in ellipse 4) + Pr(in ellipse 5)`. (See appendix 2 for advanced techniques).
# 
# For validation, we didn't use typical k-folds CV. Instead we created synthetic data and optimized our technique on synthetic data. This has proven to be more reliable than CV. Also it allows our model to use all 1024 rows of sub datasets when building models. Our model has demonstrated that on average it can perfectly classify Sklearn's `make_classification` data. However many other participants can too. So randomness is added which allows us to do better than perfect sometimes. Then two random versions' output were submitted to Kaggle and hopefully those are the high scoring ones! :P
# 
# When the code below is run locally, it generates synthetic data and calculates validation AUC. When the code is submitted to Kaggle, it uses real data and predicts `test.csv`.
# 
# [1]: https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969

# In[22]:


# RUN LOCALLY AND VALIDATE
models = 512
RunLocally = True

# RUN SUBMITTED TO KAGGLE
if len(df_test)>512*300: 
    repeat = 1
    models = 512 * repeat
    RunLocally = False
    
# INITIALIZE
all_preds = np.zeros(len(df_test))
all_y_pu = np.array([])
all_y_pr = np.array([])
all_preds_pu = np.array([])
all_preds_pr = np.array([])

# MODEL AND PREDICT
for k in range(models):
    
    # IF RUN LOCALLY AND VALIDATE
    # THEN USE SYNTHETIC DATA
    if RunLocally:
        obs = 512
        X, y = make_classification(n_samples=1024, n_features=useful[k%512], 
                    n_informative=useful[k%512], n_redundant=0, n_repeated=0,
                    n_classes=2, n_clusters_per_class=3, weights=None, flip_y=0.05,
                    class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True,
                    random_state=None)
        
    # IF RUN SUBMITTED TO KAGGLE
    # THEN USE REAL DATA
    else:
        df_train2 = df_train[df_train['wheezy-copper-turtle-magic']==k%512]
        df_test2 = df_test[df_test['wheezy-copper-turtle-magic']==k%512]
        sel = VarianceThreshold(1.5).fit(df_train2.iloc[:,1:-1])
        df_train3 = sel.transform(df_train2.iloc[:,1:-1])
        df_test3 = sel.transform(df_test2.iloc[:,1:])
        obs = df_train3.shape[0]
        X = np.concatenate((df_train3,df_test3),axis=0)
        y = np.concatenate((df_train2['target'].values,np.zeros(len(df_test2))))
    
    # TRAIN AND TEST DATA
    train = X[:obs,:]
    train_y = y[:obs]
    test = X[obs:,:]
    test_y = y[obs:]
    comb = X
                
    # FIRST MODEL : QDA
    clf = QuadraticDiscriminantAnalysis(priors = [0.5,0.5])
    clf.fit(train,train_y)
    test_pred = clf.predict_proba(test)[:,1]
    
    # SECOND MODEL : PSEUDO LABEL + QDA
    test_pred = test_pred > np.random.uniform(0,1,len(test_pred)) #randomness
    clf = QuadraticDiscriminantAnalysis(priors = [0.5, 0.5])
    clf.fit(comb, np.concatenate((train_y,test_pred)) )
    test_pred = clf.predict_proba(test)[:,1]
   
    # THIRD MODEL : PSEUDO LABEL + GAUSSIAN MIXTURE
    test_pred = test_pred > np.random.uniform(0,1,len(test_pred)) #randomness
    all_y = np.concatenate((train_y,test_pred))
    least = 0; ct = 1; thx=150
    while least<thx:
        # STOPPING CRITERIA
        if ct>=10: thx -= 10
        else: thx = 150
        # FIND CLUSTERS 
        clusters = np.zeros((len(comb),6))
        # FIND THREE TARGET=1 CLUSTERS
        train4 = comb[ all_y==1, :]
        clf = GaussianMixture(n_components=3).fit(train4) #randomness
        clusters[ all_y==1, 3:] = clf.predict_proba(train4)
        # FIND THREE TARGET=0 CLUSTERS
        train4 = comb[ all_y==0, :]
        clf = GaussianMixture(n_components=3).fit(train4) #randomness
        clusters[ all_y==0, :3] = clf.predict_proba(train4)
        # ADJUST CLUSTERS (EXPLAINED IN KERNEL COMMENTS)
        for j in range(5): clusters[:,j+1] += clusters[:,j]
        rand = np.random.uniform(0,1,clusters.shape[0])
        for j in range(6): clusters[:,j] = clusters[:,j]>rand #randomness
        clusters2 = 6 - np.sum(clusters,axis=1)
        # IF IMBALANCED TRY AGAIN 
        least = pd.Series(clusters2).value_counts().min(); ct += 1

    # FOURTH MODEL : GAUSSIAN MIXTURE + QDA
    clf = QuadraticDiscriminantAnalysis(priors = [0.167, 0.167, 0.167, 0.167, 0.167, 0.167])
    clf.fit(comb,clusters2)
    pds = clf.predict_proba(test)
    test_pred = pds[:,3]+pds[:,4]+pds[:,5]
        
    # IF RUN LOCALLY, STORE TARGETS AND PREDS
    if RunLocally:
        all_y_pu = np.append(all_y_pu, test_y[:256])
        all_y_pr = np.append(all_y_pr, test_y[256:])
        all_preds_pu = np.append(all_preds_pu, test_pred[:256])
        all_preds_pr = np.append(all_preds_pr, test_pred[256:])
    # IF RUN SUBMIT TO KAGGLE, PREDICT TEST.CSV
    else:
        all_preds[df_test2.index] += test_pred / repeat
        
    # PRINT PROGRESS
    if ((k+1)%64==0)|(k==0): print('modeled and predicted',k+1,'magic sub datasets')

# IF RUN LOCALLY, COMPUTE AND PRINT VALIDATION AUCS
if RunLocally: 
    all_y_pu_pr = np.concatenate((all_y_pu,all_y_pr))
    all_preds_pu_pr = np.concatenate((all_preds_pu,all_preds_pr))
    auc1 = roc_auc_score(all_y_pu_pr, all_preds_pu_pr)
    auc2 = roc_auc_score(all_y_pu, all_preds_pu)
    auc3 = roc_auc_score(all_y_pr, all_preds_pr)
    print()
    print('Validation AUC =',np.round(auc1,5))
    print('Approx Public LB =',np.round(auc2,5))
    print('Approx Private LB =',np.round(auc3,5))


# # Submit to Kaggle
# Alright, let's cross our fingers and hope that our submission has a high private LB !!

# In[23]:


sub = pd.read_csv('../input/instant-gratification/sample_submission.csv')
sub['target'] = all_preds
sub.to_csv('submission.csv',index=False)

plt.hist( test_pred ,bins=100)
plt.title('Model 512 test predictions')
plt.show()


# # Appendix 1 - EDA revealing n_clusters_per_class = 3
# We believe the data was made from Sklearn's `make_classification`. An important question is what parameter for `n_clusters_per_class` did Kaggle use? 
# 
# According to Sklearn's documentation [here][1]:
# 
# > This initially creates clusters of points normally distributed (std=1) about vertices of an n-informative-dimensional hypercube with sides of length 2 * class_sep and assigns an equal number of clusters to each class.
# 
# In three dimensions that means, that the clusters will be centered at one of these 8 locations: (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1) where you replace all 1's by `class_sep`. If you create 1024 rows of data and have 2 clusters per class, then for `target=1`, you may have 256 points centered at (-1, 1, -1) and 256 points centered at (1, 1, -1). Then for `target=0`, you may have 256 points centered at (1, 1, 1) and 256 points centered at (-1, -1, 1).
# 
# Using EDA, we can determine the number of clusters per class of the real data. Sklearn's `make_classification` generates data (ellipses) at hypercube corners. Therefore if there is only `n_clusters_per_class=1` then the center of each ellipse (target=1 and target=0) of data will have all coordinates 1's and -1's, for example (1,1,-1,1,-1,1,...). So if we plot a histogram of all the variables' means (center coordinates), we will see a bump at 1 and -1. (By variable we mean within each sub dataset. That's 768 rows of train and public test. We don't mean all 262144 rows of original train columns).
# 
# If `n_clusters_per_class=2`, then within one sub dataset there will be 2 ellipses for target=1. For example, there may be one ellipse centered at (-1,1,...) and one at (1,1,...) and the first coordinates of 1 and -1 will average to 0 when we compute that variable's mean. Therefore, if `clusters=2`, we will see histogram bumps at -1, 0, and 1. If `n_clusters_per_class=3`, we will see 4 bumps. Etc, etc. We can confirm this with synthetic data. We will use `n_samples=768` because we only have training and public test data to compare with which only has 768 rows per `wheezy-magic` sub dataset.
# 
# Afterward, we will plot a histogram of the real data's variable means (within sub datasets) and see which `n_clusters_per_class` it matches. Alternatively, we can build a model and assume that `n_clusters_per_class` equals 1, 2, 3, or 4. Then we check which `n_clusters_per_class` has the greatest CV. Both of these methods determine that `n_clusters_per_class=3`.
# 
# [1]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

# In[27]:


for clusters in range(4):
    centers = np.array([])
    for k in range(512):
        X, y = make_classification(n_samples=768, n_features=useful[k], 
                    n_informative=useful[k], n_redundant=0, n_repeated=0, 
                    n_classes=2, n_clusters_per_class=clusters+1, weights=None, 
                    flip_y=0.05, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, 
                    shuffle=True, random_state=None)
        centers = np.append(centers,np.mean(X[ np.argwhere(y==0).flatten() ,:],axis=0))
        centers = np.append(centers,np.mean(X[ np.argwhere(y==1).flatten() ,:],axis=0))
    plt.hist(centers,bins=100)
    plt.title('Variable means if clusters='+str(clusters+1))
    plt.show()


# ## Now let's plot the real data
# First we will use QDA to create pseudo labels for the public test data. Then we will plot a histogram of the variable means (data centers' coordinates) of target=0 and target=1 using all training and pseudo labeled public test data combined (768 rows per sub dataset). The plot below shows that Kaggle used `n_clusters_per_class=3`.

# In[28]:


centers = np.array([])
for k in range(512):
    
    # REAL DATA
    df_train2 = df_train[df_train['wheezy-copper-turtle-magic']==k]
    df_test2 = df_test[df_test['wheezy-copper-turtle-magic']==k]
    sel = VarianceThreshold(1.5).fit(df_train2.iloc[:,1:-1])
    df_train3 = sel.transform(df_train2.iloc[:,1:-1])
    df_test3 = sel.transform(df_test2.iloc[:,1:])
    obs = df_train3.shape[0]
    X = np.concatenate((df_train3,df_test3),axis=0)
    y = np.concatenate((df_train2['target'].values,np.zeros(len(df_test2))))
    
    # TRAIN AND TEST DATA
    train = X[:obs,:]
    train_y = y[:obs]
    test = X[obs:,:]
    test_y = y[obs:]
    comb = X
                
    # FIRST MODEL : QDA
    clf = QuadraticDiscriminantAnalysis(priors = [0.5,0.5])
    clf.fit(train,train_y)
    test_pred = clf.predict_proba(test)[:,1]
    
    # SECOND MODEL : PSEUDO LABEL + QDA
    test_pred = test_pred > np.random.uniform(0,1,len(test_pred))
    clf = QuadraticDiscriminantAnalysis(priors = [0.5, 0.5])
    clf.fit(comb, np.concatenate((train_y,test_pred)) )
    test_pred = clf.predict_proba(test)[:,1]
    
    # PSEUDO LABEL TEST DATA
    test_pred = test_pred > np.random.uniform(0,1,len(test_pred))
    y[obs:] = test_pred
    
    # COLLECT CENTER COORDINATES
    centers = np.append(centers,np.mean(X[ np.argwhere(y==0).flatten() ,:],axis=0))
    centers = np.append(centers,np.mean(X[ np.argwhere(y==1).flatten() ,:],axis=0))
    
# PLOT CENTER COORDINATES
plt.hist(centers,bins=100)
plt.title('Real Data Variable Means (match clusters=3)')
plt.show()


# # Appendix 2 - Advanced Techniques
# Since the code above already classifies Sklearn's `make_classification` on average prefectly, there is no need to improve it. However, below are some ideas that could be used if improvement was possible and/or neccessary.
# 
# 1. After building model 4's classifier, you could use it to classify the training data. Then all training data with `abs(oof - true)>0.9` are erroneous training data with their labels flippped. Next correct those training labels and run the entire kernel a second time.
# 
# 2. Since each cluster is centered at a hypercube corner, you can modify Sklearn's Quadratic Discriminant Analysis code by adding `meang[ np.argwhere(meang>=0) ] = 1.0` and `meang[ np.argwhere(meang<0) ] = -1.0`. This moves the centers of all clusters to hypercube corners.
# 
# 3. Computer accuracy cannot distinguish between predictions that are close to 1. Using an example with 6 digit accuracy, the numbers 1.000001 and 1.000002 are the same because both become 1.00000. To help improve AUC, you can add the following code to this kernel. `temp = np.log(pds[:,0]+pds[:,1]+pds[:,2])`; `temp[ np.isinf(temp) ] = -1e6`; `test_pred -= temp`. This improves AUC by differentiating between predictions close to 1. Note that this isn't a problem for predictions close to 0 because the numbers 0.0000001 and 0.0000002 are 1.0e-7 and 2.0e-7 and the computer can already differentiate them.
# 
# 4. After making predictions for `test.csv`, you can use them as pseudo labels and run the entire kernel a second time. Then use those labels and run the kernel a third time. Each iteration can give a slight boost.
# 
# 5. You can run this kernel multiple times and take an average. Or use k-folds. This removes this code's variance (randomness) and achieves close to prefection everytime but it also removes the possibility of scoring LB 0.00050 more or less than perfection.
# 
# 6. We can also remove this code's variance (randomness) by modifying Sklearn's code for Quadratic Discriminant Analysis and Gaussian Mixture. Each of these models can only accept training labels that are either 0 or 1. By adding a few lines of code, we can allow these models to accept continuous probabilities and use them as weights. This would allow us to remove the randomization line `test_pred = test_pred > np.random.uniform(0,1,len(test_pred))`. Instead we can leave pseudo labels as probabilities between 0 and 1 and still call `QuadraticDiscriminantAnalysis.fit(test_data,test_pred)`.
# 
# Using combinations of these additional advanced techniques, this kernel was able to score LB 0.97489 on this competition's public leaderboard. But validation showed that these techniques didn't make the basic perfect classifier any more perfect. Therefore for final submission, the basic classifier was used.

# # Appendix 3 - Final Submission Strategy
# By applying our model to synthetic data, we can learn how it performs on a simulated public and private leaderboard. We observe that this kernel achieves perfection on average (if Kaggle used `make_classification` with the parameters we suspect). Sklearn's code for `make_classification` includes
# 
#         # Randomly replace labels
#         if flip_y >= 0.0:
#             flip_mask = generator.rand(n_samples) < flip_y
#             y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())
#             
# Before variable `y` gets rewritten, you can store it by adding `y_orig = y.copy()`. Next update the shuffle line to `X, y, y_orig = util_shuffle(X, y, y_orig, random_state=generator)`. Then change the last line of `make_classification` to `return X, y, y_orig`. By doing this, we can compute the AUC of a perfect classifier with `prefect = roc_auc_score(y, y_orig)`.
#  
# Now we can make hundreds of synthetic datasets that are the similar to this competition's data and apply this kernel to see how well it does compared with a perfect classifier. For each synthetic dataset, we will run this kernel 10 times. This will show us patterns and help us decide how to choose our two final submissions. 
#  
# We observe that sometimes this kenel does better than perfect and sometimes it does worse than perfect. It's interesting to note that there is no correlation between its performance on public LB versus private LB. In the example plots below, perfect classification is represented by the black dotted line.
# 
# If we take the average of this kernel's performance over many sythnetic datasets, it achieves perfect classification. Therefore there is no reason to improve this kernel. We can never expect to perform better than perfect classification on average. The only change we could consider is altering this kernel's standard deviation from perfection. We could either try to achieve perfection every kernel run, or (leave it as is and) randomly try to exceed perfection by a desired amount on some kernel runs.
#  
# ## Synthetic Dataset 1
# ![image](http://playagricola.com/Kaggle/ex161919.png)
# ## Synthetic Dataset 2
# ![image](http://playagricola.com/Kaggle/ex261919.png)
# ## Synthetic Dataset 3
# ![image](http://playagricola.com/Kaggle/ex361919.png)
# ## Synthetic Dataset 4
# ![image](http://playagricola.com/Kaggle/ex461919.png)
# ## Synthetic Dataset 5
# ![image](http://playagricola.com/Kaggle/ex561919.png)
# ## Many Synthetic Datasets Together
# ![image](http://playagricola.com/Kaggle/scatter61719.png)
