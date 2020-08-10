import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/predict-who-is-more-influential-in-a-social-network/train.csv")
test = pd.read_csv("../input/predict-who-is-more-influential-in-a-social-network/test.csv")


def pre_pro(df):
    df = df.astype('float32')
    col = df.columns
    for i in range(len(col)):
        m = df.loc[df[col[i]] != -np.inf, col[i]].min()
        df[col[i]].replace(-np.inf, m, inplace=True)
        M = df.loc[df[col[i]] != np.inf, col[i]].max()
        df[col[i]].replace(np.inf, M, inplace=True)

    df.fillna(0, inplace=True)
    return df


def feat_eng(df):
    df.replace(0, 0.001)

    df['follower_diff'] = (df['A_follower_count'] > df['B_follower_count'])
    df['following_diff'] = (df['A_following_count'] > df['B_following_count'])
    df['listed_diff'] = (df['A_listed_count'] > df['B_listed_count'])
    df['ment_rec_diff'] = (df['A_mentions_received'] > df['B_mentions_received'])
    df['rt_rec_diff'] = (df['A_retweets_received'] > df['B_retweets_received'])
    df['ment_sent_diff'] = (df['A_mentions_sent'] > df['B_mentions_sent'])
    df['rt_sent_diff'] = (df['A_retweets_sent'] > df['B_retweets_sent'])
    df['posts_diff'] = (df['A_posts'] > df['B_posts'])

    df['A_pop_ratio'] = df['A_mentions_sent'] / df['A_listed_count']
    df['A_foll_ratio'] = df['A_follower_count'] / df['A_following_count']
    df['A_ment_ratio'] = df['A_mentions_sent'] / df['A_mentions_received']
    df['A_rt_ratio'] = df['A_retweets_sent'] / df['A_retweets_received']

    df['B_pop_ratio'] = df['B_mentions_sent'] / df['B_listed_count']
    df['B_foll_ratio'] = df['B_follower_count'] / df['B_following_count']
    df['B_ment_ratio'] = df['B_mentions_sent'] / df['B_mentions_received']
    df['B_rt_ratio'] = df['B_retweets_sent'] / df['B_retweets_received']

    df['A/B_foll_ratio'] = (df['A_foll_ratio'] > df['B_foll_ratio'])
    df['A/B_ment_ratio'] = (df['A_ment_ratio'] > df['B_ment_ratio'])
    df['A/B_rt_ratio'] = (df['A_rt_ratio'] > df['B_rt_ratio'])

    df['nf1_diff'] = (df['A_network_feature_1'] > df['B_network_feature_1'])
    df['nf2_diff'] = (df['A_network_feature_2'] > df['B_network_feature_2'])
    df['nf3_diff'] = (df['A_network_feature_3'] > df['B_network_feature_3'])

    df['nf3_ratio'] = df['A_network_feature_3'] / df['B_network_feature_3']
    df['nf2_ratio'] = df['A_network_feature_2'] / df['B_network_feature_2']
    df['nf1_ratio'] = df['A_network_feature_1'] / df['B_network_feature_1']

    return (pre_pro(df))

feat_train = feat_eng(train.copy())
feat_test = feat_eng(test.copy())
y_train = np.array(feat_train['Choice'])
feat_train.drop('Choice', axis=1, inplace = True)

XGB = xgb.XGBRegressor(max_depth=5, n_estimators=100, colsample_bytree=0.4, subsample=0.95, nthread=10,
                       learning_rate=0.01, gamma=0.045,min_child_weight=1.5,reg_alpha=0.65,reg_lambda=0.45)
XGB.fit(feat_train, y_train)

pred1 = XGB.predict(feat_test)
pred1 = (pred1-min(pred1))/(max(pred1)-min(pred1))

regress = RandomForestRegressor(max_depth=10, random_state=None, n_estimators=1000)
regress.fit(feat_train, y_train)
pred2 = regress.predict(feat_test)
pred2 = (pred2-min(pred2))/(max(pred2)-min(pred2))

bayesian = BayesianRidge(n_iter=6000, tol=0.01, alpha_1=0.01,
alpha_2=0.2, lambda_1=0.01, lambda_2=0.02, alpha_init=None, lambda_init=None, 
compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)

bayesian.fit(feat_train, y_train)
pred3 = bayesian.predict(feat_test)
pred3 = (pred3-min(pred3))/(max(pred3)-min(pred3))


lgb = LGBMRegressor(n_estimators = 1000, learning_rate = 0.1, max_depth = 2)
lgb.fit(feat_train, y_train)
pred4 = lgb.predict(feat_test)
pred4 = (pred4-min(pred4))/(max(pred4)-min(pred4))

pred = 0.25*pred1+0.1*pred2+0.05*pred3+0.6*pred4
submission = pd.read_csv('../input/predict-who-is-more-influential-in-a-social-network/sample_predictions.csv')
submission['Choice'] = pred
submission.to_csv('sub.csv', index = False, header = True)
