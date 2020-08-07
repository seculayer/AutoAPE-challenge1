import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from lightgbm import LGBMClassifier
print(os.listdir("../input"))

train = pd.read_csv("../input/airbnb-new-files/train_users_2.csv")
test = pd.read_csv("../input/airbnb-new-files/test_users.csv")

y_train = train['country_destination'].values
le = LabelEncoder()
y_train = le.fit_transform(y_train)

train = train.drop(['id','country_destination','date_first_booking'], axis=1)
test = test.drop(['id','date_first_booking'], axis=1)


train_size = train.shape[0]

data_all = pd.concat((train, test), axis=0, ignore_index = True)
data_all = data_all.fillna(-1)


dac = np.vstack(data_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
data_all['dac_year'] = dac[:,0]
data_all['dac_month'] = dac[:,1]
data_all['dac_day'] = dac[:,2]
data_all = data_all.drop(['date_account_created'], axis=1)

tfa = np.vstack(data_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
data_all['tfa_year'] = tfa[:,0]
data_all['tfa_month'] = tfa[:,1]
data_all['tfa_day'] = tfa[:,2]
data_all = data_all.drop(['timestamp_first_active'], axis=1)

data_all.loc[data_all['age']<14] = 0
data_all.loc[data_all['age']>100] = 0

categorical_columns = ['affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method',
    'signup_flow']

for col in categorical_columns:
    data_all_dummy = pd.get_dummies(data_all[col], prefix=col)
    data_all = data_all.drop([col], axis=1)
    data_all = pd.concat((data_all, data_all_dummy), axis=1)

vals = data_all.values
train = vals[:train_size]
test = vals[train_size:]


lgb = LGBMClassifier(max_depth=10, learning_rate=0.01, n_estimators=1000,
                    objective='multi:softprob',subsample=0.4, colsample_bytree=0.6, eval_metric='mlogloss')
lgb.fit(train, y_train)
pred = lgb.predict_proba(test)


submission = pd.read_csv('../input/airbnb-new-files/sample_submission_NDF.csv')
id_sub = submission['id']

ids = []
cts = []
for i in range(len(id_sub)):
    idx = id_sub[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(pred[i])[::-1])[:5].tolist()

submission = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
submission.to_csv('sub_airbnb.csv', index=False)
