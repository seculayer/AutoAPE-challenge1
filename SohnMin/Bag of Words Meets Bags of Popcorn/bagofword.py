

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup    
from nltk.corpus import stopwords 


train = pd.read_csv("D:/kaggle_data/word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
train.shape
train.head()
train.info()


#추가 데이터
imdb = pd.read_csv("D:/kaggle_data/word2vec-nlp-tutorial/imdb_master.csv", encoding = "ISO-8859-1")
imdb.head()
imdb=imdb.drop(["Unnamed: 0","type","file"], axis=1)
imdb['sentiment']=imdb['label']
imdb=imdb.drop(["label"], axis=1)

#sentiment의 neg pos 를 0과 1로
def senti(sentiment):
    if sentiment =='neg':
        return 0
    else:
        return 1

    

imdb['sentiment']=imdb.apply(lambda x:senti(x['sentiment']),axis=1)
imdb.drop_duplicates(keep="last",inplace=True)

imdb.head()

all=pd.concat([train,imdb],axis=0,  ignore_index = True)

all.shape

y=all['sentiment']

all=all.drop(['sentiment'],axis=1)

all.shape

test = pd.read_csv("D:/kaggle_data/word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", quoting=3)

all=pd.concat([all,test],axis=0,  ignore_index = True)

all.shape

all.head()



def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words= letters_only.lower().split()
    words = [w for w in words if not w in stopwords.words("english")]
    return( ' '.join(words))


print(all["review"].size)


num_reviews = all["review"].size
clean_reviews = []
for i in range( 0, num_reviews ):
    if( (i+1)%10000 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_reviews )) 
    clean_reviews.append( review_to_words( all["review"][i] ))



print(len(clean_reviews))


# TF-iDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features= 10000)
all_review = tfidf.fit_transform(clean_reviews)
all_review=all_review.toarray()

print(len(y))


X_train=all_review[:len(y)]

#LGBM 사용
from lightgbm import LGBMClassifier


lgbm_model = LGBMClassifier(n_estimators=220, learning_rate=0.2, num_leaves=120, random_state=77)
lgbm_model.fit(X_train,y)


X_test = all_review[len(y):]
preds = lgbm_model.predict(X_test)



submit=pd.DataFrame(all["id"][len(y):])
submit['sentiment']=preds
submit.head()


#id의 ""를 지우는 함수
def remove_d(word):
    word=word.replace('"',"")
    return word


submit['id']=submit.apply(lambda x:remove_d(x['id']), axis=1)

submit.head()
submit.to_csv("Bag_of_Words_model_LGB_imdb.csv", index=False, quoting=3 )

