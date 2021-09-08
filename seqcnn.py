# * Comorasu Ana-Maria
# * Grupa 234
# * Sequential CNN Model

# * imports

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from PIL import ImageFile
from keras import backend as bk
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle

from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, SeparableConv2D, LayerNormalization
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler

from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import LearningRateScheduler

import json
import numpy as np
import pandas as pd

import PIL
import PIL.Image

from keras.callbacks import EarlyStopping


# ? unzip files
# ! files should be in directories corresponding to their classes
local_zipfile='/content/drive/MyDrive/MachineLearning/CT_SCAN_DATASET.zip'
zip_ref =zipfile.ZipFile(local_zipfile, 'r')
zip_ref.extractall()
zip_ref.close()

# * images for train and validation
training_data = ImageDataGenerator(
    rotation_range=20.,
    data_format='channels_last'
)
training_generator = training_data.flow_from_directory(
    '/content/CT_SCAN_DATASET/train_folder/',
    target_size = (50, 50),
    batch_size = 64,
    class_mode='categorical'
)

validation_data = ImageDataGenerator(
    data_format='channels_last'
)

validation_generator = validation_data.flow_from_directory(
    '/content/CT_SCAN_DATASET/validation_folder/',
    target_size = (50, 50),
    batch_size = 64,
    class_mode='categorical'
)

# * Sequential Model

model = Sequential()
model.add(BatchNormalization(input_shape=(50, 50, 3)))

model.add(Conv2D(64, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(512, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer=Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# * Fit model
history = model.fit(training_generator, epochs=250, validation_data = validation_generator, verbose = 1)

# * Evaluate Model
model.evaluate(validation_generator)

# * Make submission on Kaggle 
data_file = open("test.txt")
with open("seq_submission.txt", "w") as f:
  for im in data_file:
    im = im.rstrip()
    path = "/content/CT_SCAN_DATASET/test/" + im
    img = image.load_img(path, target_size=(50, 50))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.hstack([x])
    classes = model.predict(images)
    if (classes[0][0]>classes[0][1]) and (classes[0][0]>classes[0][2]):
      f.write(im+",0\n")

    elif classes[0][1]>classes[0][2]:
      f.write(im+",1\n")
    else:
      f.write(im+",2\n")

# * Save model if needed
model.save("modelseq.h5")

# * Load model if needed
model=tf.keras.models.load_model("/content/modelseq.h5")