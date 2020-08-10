# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




get_ipython().system('pip install pyunpack')
get_ipython().system('pip install patool')

os.system('apt-get install p7zip')

#압축파일 해제
import os
from pyunpack import Archive
import shutil
if not os.path.exists('/kaggle/working/mercari-price-suggestion-challenge/'):
    os.makedirs('/kaggle/working/mercari-price-suggestion-challenge/')
Archive('/kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z').extractall('/kaggle/working/mercari-price-suggestion-challenge/')
for dirname, _, filenames in os.walk('/kaggle/working/mercari-price-suggestion-challenge/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


if not os.path.exists('/kaggle/working/mercari-price-suggestion-challenge/'):
    os.makedirs('/kaggle/working/mercari-price-suggestion-challenge/')
Archive('/kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip').extractall('/kaggle/working/mercari-price-suggestion-challenge/')
for dirname, _, filenames in os.walk('/kaggle/working/mercari-price-suggestion-challenge/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd 
import numpy as np
import gc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer


train = pd.read_csv('/kaggle/working/mercari-price-suggestion-challenge/train.tsv',sep="\t")
train.head()

test = pd.read_csv('/kaggle/working/mercari-price-suggestion-challenge/test_stg2.tsv',sep="\t")
test.head()



train.head()




os.remove('/kaggle/working/mercari-price-suggestion-challenge/test_stg2.tsv')
os.remove('/kaggle/working/mercari-price-suggestion-challenge/train.tsv')



#결측치 처리
train["category_name"]=train["category_name"].fillna(value="empty/empty/empty")
test["category_name"]=test["category_name"].fillna(value="empty/empty/empty")
train["brand_name"]=train["brand_name"].fillna(value="empty")
test["brand_name"]=test["brand_name"].fillna(value="empty")
train["item_description"]=train["item_description"].fillna(value="No description yet")
test["item_description"]=test["item_description"].fillna(value="No description yet")



#카테고리 세분화
def split_cat(category_name):
        return category_name.split('/')

train['cat_first'], train['cat_second'], train['cat_third'] = zip(*train['category_name'].apply(lambda x : split_cat(x)))
test['cat_first'], test['cat_second'], test['cat_third'] = zip(*test['category_name'].apply(lambda x : split_cat(x)))
train=train.drop(['category_name'], axis=1)
test=test.drop(['category_name'],axis=1)



#정규 분포화
y = np.log1p(train.price)

train = train.drop('price', axis=1)
train.rename(columns={'train_id': 'id'}, inplace=True)
test.rename(columns={'test_id': 'id'}, inplace=True)
all: pd.DataFrame = pd.concat([train, test])
del train 
del test
gc.collect()



#카운터벡터
cnt = CountVectorizer()
X_name = cnt.fit_transform(all.name)



#tf-idf
tfidf = TfidfVectorizer(max_features= 50000,ngram_range=(1, 2), stop_words='english')
X_description = tfidf.fit_transform(all['item_description'])



#원핫인코딩
OH_catfirst = LabelBinarizer(sparse_output=True)
X_catfirst = OH_catfirst.fit_transform(all['cat_first'])



OH_catsecond = LabelBinarizer(sparse_output=True)
X_catsecond = OH_catsecond.fit_transform(all['cat_second'])



OH_catthird = LabelBinarizer(sparse_output=True)
X_catthird = OH_catthird.fit_transform(all['cat_third'])




OH_brandname = LabelBinarizer(sparse_output=True) 
X_brandname = OH_brandname.fit_transform(all['brand_name'])




OH_shipping = LabelBinarizer(sparse_output=True)
X_shipping = OH_shipping.fit_transform(all['shipping'])




OH_itemconditionid = LabelBinarizer(sparse_output=True)
X_itemconditionid = OH_itemconditionid.fit_transform(all['item_condition_id'])


#메모리 해제
all.head()



all=all[['id','item_description']]

gc.collect()


all.head()



#행렬 결합
from scipy.sparse import hstack


all_matrix = (X_name, X_description,X_catfirst, X_catsecond, X_catthird, X_brandname, X_shipping, X_itemconditionid )
X_all_feature = hstack(all_matrix).tocsr()



#메모리 해제
all=all['id']

submit = all[len(y):]
del all_matrix
del all
gc.collect()




X_train=X_all_feature[:len(y)]



#LGBM 사용
from lightgbm import LGBMRegressor


lgbm_model = LGBMRegressor(n_estimators=250, learning_rate=0.2, num_leaves=150, random_state=77)
lgbm_model.fit(X_train,y)




X_test = X_all_feature[len(y):]
preds = lgbm_model.predict(X_test)



submit=pd.DataFrame(submit)
submit["test_id"]=submit["id"]
submit["price"]=np.expm1(preds)
submit[["test_id","price"]].to_csv("submission.csv",index=False)

