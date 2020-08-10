import pylab as pl
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/GiveMeSomeCredit/cs-training.csv")
test = pd.read_csv("../input/GiveMeSomeCredit/cs-test.csv")

train = train.dropna()
train = train.drop(columns = ['Unnamed: 0'])

y_ans = train['SeriousDlqin2yrs']
feat_train = train.drop('SeriousDlqin2yrs',axis=1)
test = test.drop(columns=['Unnamed: 0'])
feat_test = test.drop('SeriousDlqin2yrs', axis=1)

feat_train = feat_train.fillna(-1)
test = test.fillna(-1)

XGB = xgb.XGBRegressor(max_depth=7, n_estimators=1000, colsample_bytree=0.45, subsample=0.95, nthread=10,
                       learning_rate=0.00001, gamma=0.05,min_child_weight=1.5,reg_alpha=0.65,reg_lambda=0.45)
XGB.fit(feat_train, y_ans)
pred = XGB.predict(feat_test)
pred = (pred-min(pred))/(max(pred)-min(pred))

lgb = LGBMRegressor(n_estimators = 1000, learning_rate = 0.001, max_depth = 7)
lgb.fit(feat_train, y_ans)
pred2 = lgb.predict(feat_test)
pred2 = (pred2-min(pred2))/(max(pred2)-min(pred2))

pred = pred*0.65+pred2*0.35

submission = pd.read_csv('../input/GiveMeSomeCredit/sampleEntry.csv')
submission['Probability'] = pred
submission.to_csv('sub_credit.csv', index = False, header = True)
