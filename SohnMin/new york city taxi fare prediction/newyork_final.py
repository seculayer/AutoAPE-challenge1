import pandas as pd
import numpy as np
import datetime as dt

train=pd.read_csv('D:/kaggle_data/new-york-city-taxi-fare-prediction/train.csv')
test=pd.read_csv('D:/kaggle_data/new-york-city-taxi-fare-prediction/test.csv')

train.head()

train=train.dropna(axis=0)
train=train[train.fare_amount>0] #비정상적인 요금 값 제거
train=train[train.passenger_count>0] #비정상적인 승객 수 제거
train=train[5>train.passenger_count]


#거리를 구하는 함수
from math import sin, cos, radians, atan2,acos
def distance(x1,y1,x2,y2):
    x1=radians(float(x1))
    y1=radians(float(y1))
    x2=radians(float(x2))
    y2=radians(float(y2))
    z=sin(x1)*sin(x2)+cos(x1)*cos(x2)*cos(y1-y2)
    if(z>1):   #값이 가끔 1을 넘김
        z=1
    R =6371.01
    return R*acos(z)


train['distance']=train.apply(lambda x:distance(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']),axis=1)
train=train.drop(["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"],1)



train=train[(train.distance>0.3)] #거리가 작은값 제거
train=train[(train.fare_amount<100)&(train.distance<40)] #비정상값 제거


#날짜를 쪼개는 함수
def add_datetime_info(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")  
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year
    
    return dataset



train=add_datetime_info(train)

test['distance']=test.apply(lambda x:distance(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']),axis=1)
test=add_datetime_info(test)


#XGB사용
import xgboost as xgb 


model = xgb.XGBRegressor(learning_rate=0.1,max_depth=5, n_estimators=100) 

X= train[['distance','hour','day','month','year','weekday']]
y= train[['fare_amount']]

model.fit(X,y)

test['fare_amount']=model.predict(test[['distance','hour','day','month','year','weekday']])

submission=test[['key', 'fare_amount']]
submission.to_csv('xgb_date_submission.csv', index=False)





