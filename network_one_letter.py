import random
import os
import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import PIL
import PIL.Image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import pathlib

imgSize = 28
classCount = 26
batch_size = 256

AUTOTUNE = tf.data.AUTOTUNE

train_dir = pathlib.Path('dataset/train/one')

list_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'), shuffle=True)
# list_ds = tf.data.Dataset.list_files(str(train_dir/'*/*'), shuffle=True).concatenate(tf.data.Dataset.list_files(str(train_dir2/'*/*'), shuffle=True)).concatenate(tf.data.Dataset.list_files(str(train_dir3/'*/*'), shuffle=True)).concatenate(tf.data.Dataset.list_files(str(train_dir4/'*/*'), shuffle=True))
image_count = len(list_ds)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=True)
train_ds = list_ds.take(image_count)

val_dir = pathlib.Path('dataset/validation/one')
list_ds2 = tf.data.Dataset.list_files(str(val_dir/'*/*'), shuffle=True)
val_image_count = len(list_ds2)
list_ds2 = list_ds2.shuffle(val_image_count, reshuffle_each_iteration=True)
val_ds = list_ds2.take(val_image_count)

# val_size = int(image_count * 0.2)
# train_ds = list_ds.skip(val_size)
# val_ds = list_ds.take(val_size)

# val_ds = list_ds.take(0)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

# class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
print(class_names)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    return one_hot

def decode_img(img):
    # convert the compressed string to a 1D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    img = img / 255
    # resize the image to the desired size
    # return tf.image.resize(img, [imgSize, imgSize])
    return img

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in train_ds.take(1):
#     print("Image shape: ", image.numpy())
#     print("Label: ", label.numpy())

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

inputs = tf.keras.Input(shape=(imgSize,imgSize,1))
model1 = tf.keras.Sequential([ 
    inputs,
    # tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu',input_shape=(imgSize,imgSize,1)),
    tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(classCount, activation='softmax')
])

model1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# model1 = tf.keras.models.load_model('model/one_letter/my_model')
 
MCP = ModelCheckpoint('model/one_letter/Best_points.h5',verbose=1,save_best_only=True,monitor='val_accuracy',mode='max')
ES = EarlyStopping(monitor='val_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=5,mode='max')
RLP = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.2,min_lr=0.0001)

model1.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[MCP,ES,RLP])

# evaluate the model
scores = model1.evaluate(train_ds)
print("Train: %s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))
scores = model1.evaluate(val_ds)
print("Test: %s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))

model1.save("model/one_letter/my_model")
# model1.save("model/one_letter/my_model2")
