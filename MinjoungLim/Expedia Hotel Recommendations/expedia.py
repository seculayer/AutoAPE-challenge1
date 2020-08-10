import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
import os


print(os.listdir("../input"))

train = pd.read_csv("../input/expedia-hotel-recommendations/train.csv",nrows=100000)
test = pd.read_csv("../input/expedia-hotel-recommendations/test.csv")

def datetime(data, column):
    data[column] = pd.to_datetime(data[column],errors='coerce')
    year = data[column].dt.year
    month = data[column].dt.month
    day = data[column].dt.day
    return year, month, day

train['dt_year'],train['dt_month'],train['dt_day'] = datetime(train,'date_time')
train['ci_year'],train['ci_month'],train['ci_day'] = datetime(train,'srch_ci')
train['co_year'],train['co_month'],train['co_day'] = datetime(train,'srch_co')

test['dt_year'],test['dt_month'],test['dt_day'] = datetime(test,'date_time')
test['ci_year'],test['ci_month'],test['ci_day'] = datetime(test,'srch_ci')
test['co_year'],test['co_month'],test['co_day'] = datetime(test,'srch_co')

train = train.drop(['date_time','srch_ci','srch_co','user_id','is_mobile','is_booking','cnt'],axis=1)
test = test.drop(['date_time','srch_ci','srch_co','id','user_id','is_mobile'],axis=1)

y_train = train['hotel_cluster'].values
train = train.drop('hotel_cluster', axis=1)

xgb = XGBClassifier(num_class=12 ,max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob',subsample=0.4, colsample_bytree=0.5, eval_metric='mlogloss')
xgb.fit(train, y_train)
pred = xgb.predict_proba(test)

submission = pd.read_csv("../input/expedia-hotel-recommendations/sample_submission.csv")

id_sub = submission['id']

hts= [np.argsort(pred[i][::-1])[:5] for i in range(len(id_sub))]
write_p = [" ".join([str(l) for l in p]) for p in hts]
write_frame = ["{0},{1}".format(id_sub[i], write_p[i]) for i in range(len(hts))]
write_frame = ["id,hotel_cluster"] + write_frame

with open("sub_expedia.csv", "w+") as f:
    f.write("\n".join(write_frame))
