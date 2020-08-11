import glob
import math, re, os
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import operator


sub = pd.read_csv('sample_submission.csv')
# train_filenames = np.array(os.listdir("train/train/"))
test_filenames = np.array(os.listdir("test/"))
test_path = []

for testname in test_filenames:
    test_filenames
    path = os.path.join("test/", testname)
    test_path.append(path)

train_data = pd.read_csv('train.csv').to_numpy()

# print(train_data)

c = Counter()
for image, label in train_data:
    # print(image, label)
    c[label] += 1

print(c.most_common(1000))
sums = 0
for i in c.most_common(1000):
    sums = sums + i[1]

print(sums)

labels = [item[0] for item in c.most_common(4000)]
new_train_data = []
train_paths = []
train_labels = []
for data in train_data:
    if data[1] in labels:
        new_train_data.append([data[1], data[0]])
        path = os.path.join("train/", data[0])
        train_paths.append(path)
        train_labels.append(data[1])

print(new_train_data)

print(len(train_paths))
print(len(train_labels))

'''
le = LabelEncoder()
label_encode = le.fit_transform(train_labels)
print(label_encode)
label_encode = label_encode.reshape(len(label_encode), 1)
print(label_encode)

enc = OneHotEncoder()
label_encode = enc.fit_transform(label_encode)
i = 0
'''
le = LabelEncoder()
label_encode = le.fit_transform(train_labels)
print(label_encode)
label_encode = to_categorical(label_encode)
print(label_encode)
# print(train_paths[i], train_labels[i], label_encode[i])

# new whale ~ 100th labels, label vector creation

# for image, id in range(train_data):


train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, label_encode, test_size=0.20)

print(valid_paths)
print(train_labels)

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE
# Configuration
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IM_SIZE = 100


def decode_image(filename, label=None, image_size=(IM_SIZE, IM_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)

    if label is None:
        return image
    else:
        return image, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if label is None:
        return image
    else:
        return image, label


train_dataset = (
    tf.data.Dataset
        .from_tensor_slices((train_paths, train_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .cache()
        .repeat()
        .shuffle(1024)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
        .from_tensor_slices((valid_paths, valid_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
        .from_tensor_slices(test_path)
        .map(decode_image, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
)

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential


def create_model_b2():
    efficient_net = efn.EfficientNetB2(weights='imagenet', include_top=False,
                                       input_shape=(100, 100, 3), pooling='max')
    return tf.keras.Sequential(
        [
            efficient_net,
            tf.keras.layers.Dense(2000, activation='relu'),
            tf.keras.layers.Dense(4000, activation='sigmoid')
        ])

'''
with strategy.scope():
    model = create_model_b2()
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

EPOCH = 3
STEPS_PER_EPOCH = 5

checkpoint = callbacks.ModelCheckpoint('model_b2', monitor='val_loss', save_best_only=True, verbose=1, period=1)

reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', min_lr=0.00001, patience=3, mode='min', verbose=1)

print("===========training start===========")
print("train dataset: ", train_dataset)
print("epoch: ", EPOCH)
print("steps per epoch: ", STEPS_PER_EPOCH)
print("valid dataset: ", valid_dataset)

history = model.fit(train_dataset, epochs=EPOCH, steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_dataset,
                    callbacks=[checkpoint, reduceLR])

out_vector = model.predict(test_dataset, verbose=1)
'''

out_label = []
for index, out in enumerate(out_vector):
    top_five = np.empty(5, dtype=int)
    #print("index: ", index, "out: ", out)
    #top_five[0] = 0
    top_five[:] = (sorted(range(len(out)), key=lambda i: out[i])[-5:])
    #print("top five : " , top_five)
    #print(le.inverse_transform(top_five))
    out_label.append(' '.join(le.inverse_transform(top_five)))


#print(out_label)

sub['Id'] = out_label
sub.to_csv('submission.csv', index=False)
sub.head()

