import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from os.path import join
from sklearn.model_selection import train_test_split
import os

print(os.listdir("../input"))

data_dir = "../input/dog-breed-identification"
labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))

labels['dir'] = data_dir+'/train/'+labels['id']+'.jpg'
test = sample_submission['id']
test_img = data_dir+'/test/'+test+'.jpg'

targets = pd.Series(labels['breed'])
one_hot = pd.get_dummies(targets, sparse = True)
one_hot_labels = np.asarray(one_hot)

img_rows = 84
img_cols = 84

x_feature = []
y_feature = []
i = 0
for img in tqdm(labels.values):
    train_img = cv2.imread(labels['dir'][i],cv2.IMREAD_COLOR)
    label = one_hot_labels[i]
    train_img_resize = cv2.resize(train_img, (img_rows, img_cols))
    x_feature.append(train_img_resize)
    y_feature.append(label)
    i += 1

x_train_data = np.array(x_feature, np.float32) / 255.
y_train_data = np.array(y_feature, np.uint8)


x_test_feature = []
for f in tqdm(test_img.values):
    img = cv2.imread(test_img[i], cv2.IMREAD_COLOR)
    img_resize = cv2.resize(img, (img_rows, img_cols))
    x_test_feature.append(img_resize)

x_test_data = np.array(x_test_feature, np.float32) / 255.

x_train, x_val, y_train, y_val = train_test_split(x_train_data, y_train_data, test_size=0.3)

model = Sequential()
model.add(Convolution2D (filters = 64, kernel_size = (4,4),padding = 'Same',
                         activation ='relu', input_shape = (img_rows, img_cols,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D (filters = 64, kernel_size = (4,4),padding = 'Same',
                         activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units = 120, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=["accuracy"])

batch_size = 700
nb_epochs = 50
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=nb_epochs,
                    verbose=2,
                    validation_data=(x_val, y_val),
                    initial_epoch=0)

results = model.predict(x_test_data)
prediction = pd.DataFrame(results)

col_names = one_hot.columns.values
prediction.columns = col_names
prediction.insert(0, 'id', sample_submission['id'])

submission = prediction
submission.to_csv('new_submission.csv', index=False)
