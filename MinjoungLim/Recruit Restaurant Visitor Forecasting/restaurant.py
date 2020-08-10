import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn import preprocessing

import os
print(os.listdir("../input"))

data = {
    'tra': pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/air_visit_data.csv'),
    'as': pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/air_store_info.csv'),
    'hs': pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/air_reserve.csv'),
    'hr': pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/hpg_reserve.csv'),
    'id': pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/store_id_relation.csv'),
    'tes': pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/sample_submission.csv'),
    'hol': pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)
    [['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores,
                                  'dow': [i]*len(unique_stores)}) for i in range(7)],axis=0, ignore_index=True).reset_index(drop=True)

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors']\
    .min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors']\
    .mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors']\
    .median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors']\
    .max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors']\
    .count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

data['tra'] = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
data['tes'] = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(data['tra'], stores, how='left', on=['air_store_id','dow'])
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date'])
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

y_train = train['visitors']
feat_train = train.drop(['air_store_id','visit_date','visitors'], axis=1)
feat_test = test.drop(['air_store_id','visit_date','visitors','id'], axis=1)

XGB = xgb.XGBRegressor(num_class=12 ,max_depth=7, n_estimators=1000, colsample_bytree=0.4, subsample=1.0, nthread=10,
                       learning_rate=0.35, gamma=0.05,min_child_weight=1.5,reg_alpha=0.65,reg_lambda=0.45)
XGB.fit(feat_train, y_train)
pred = XGB.predict(feat_test)

regress = RandomForestRegressor(max_depth=8, random_state=None, n_estimators=1000)
regress.fit(feat_train, y_train)
pred2 = regress.predict(feat_test)

lgb = LGBMRegressor(n_estimators = 1500, learning_rate = 0.41, max_depth = 7)
lgb.fit(feat_train, y_train)
pred3 = lgb.predict(feat_test)

pred = 0.35*pred+0.1*pred2+0.55*pred3


submission = pd.read_csv('../input/recruit-restaurant-visitor-forecasting-data/sample_submission.csv')
submission['visitors'] = np.round(pred)
submission.to_csv('sub_restaurant.csv', index = False, header = True, encoding='utf-8')
