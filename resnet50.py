# * Comorasu Ana-Maria
# * Grupa 234
# * ResNet50 Model - Transfer Learning

# ? imports
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

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

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
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# ? unzip files
# ! files should be in directories corresponding to their classes
local_zipfile='/content/drive/MyDrive/MachineLearning/CT_SCAN_DATASET.zip'
zip_ref =zipfile.ZipFile(local_zipfile, 'r')
zip_ref.extractall()
zip_ref.close()

# * images for train and validation
training_data = ImageDataGenerator(
    rotation_range=20.,
    width_shift_range=0.2,
    height_shift_range=0.2, 
    zoom_range=0.2,
    data_format='channels_last'
)

training_generator = training_data.flow_from_directory(
    '/content/CT_SCAN_DATASET/train_folder/',
    target_size = (224, 224),
    batch_size = 32,
    class_mode='categorical'
)

validation_data = ImageDataGenerator(
    data_format='channels_last'
)

validation_generator = validation_data.flow_from_directory(
    '/content/CT_SCAN_DATASET/validation_folder/',
    target_size = (224, 224),
    batch_size = 32,
    class_mode='categorical'
)

# * Learning Rate Scheduler
learning_rate = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x, verbose=0)

# * Early Stopping
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=1e-3, patience=20, verbose=1, mode='max', restore_best_weights=True)

# * ResNet50 Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# * function for adding final layers
def layer_adder(bottom_model):
  top_model = bottom_model.output
  top_model = GlobalAveragePooling2D()(top_model)
  top_model = Dense(1024,activation='relu')(top_model)
  top_model = Dense(3,activation='softmax')(top_model)
  return top_model

# * add layer to model
top = layer_adder(model)
resnet_model = Model(inputs=model.input, outputs=top)

# * model summary
resnet_model.summary()

# * compile with Adam Optimizer
resnet_model.compile(optimizer=Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

# * train model
history = resnet_model.fit(training_generator, epochs=50, validation_data=validation_generator, callbacks=[learning_rate, early_stopping],verbose=1)

# * evaluate model
resnet_model.evaluate(validation_generator)

# * make submission on kaggle

data_file = open("test.txt")
with open("my_submission.txt", "w") as f:

  for im in data_file:
    im = im.rstrip()
    path = "/content/CT_SCAN_DATASET/test/" + im
    img = image.load_img(path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.hstack([x])
    classes = resnet_model.predict(images)
    if (classes[0][0]>classes[0][1]) and (classes[0][0]>classes[0][2]):
      f.write(im+",0\n")
    elif classes[0][1]>classes[0][2]:
      f.write(im+",1\n")
    else:
      f.write(im+",2\n")
