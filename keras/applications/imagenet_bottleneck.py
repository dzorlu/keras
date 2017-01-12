#!/usr/bin/python
from __future__ import absolute_import

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, InputLayer
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input, _obtain_input_shape
from keras import activations, initializations, regularizers, constraints
from keras.utils.np_utils import to_categorical

import os
import numpy as np
from collections import OrderedDict
from itertools import chain
import time

# dimensions of our images.
img_width, img_height = 299, 299
todaysdate = time.strftime("%Y_%m_%d")
TRAIN_DATA_DIR = '/Users/denizzorlu/style/data/train'
VALID_DATA_DIR = '/Users/denizzorlu/style/data/validation'
TOP_MODEL_WEIGHTS_PATH = '/Users/denizzorlu/keras/saved_models/bottleneck_fc_model_{}.h5'.format(todaysdate)
nb_epoch = 50
BATCH_SIZE = 64
SAMPLE_SIZE = BATCH_SIZE * 400
VALIDATION_FRACTION = 0.2
VAL_SAMPLE_SIZE = SAMPLE_SIZE * VALIDATION_FRACTION
NB_WORKERS = 4

def number_of_images(folder):
    class_totals = dict()
    for root, dirs, files in os.walk(folder):
        class_ = root.split("/")[-1]
        class_totals[class_] = len(files)
    return class_totals

def create_labels(dir_):
    # create labels in the same orders read by the generator - sorted
    image_counts = number_of_images(dir_)
    classes = []
    for subdir in sorted(os.listdir(dir_)):
        if os.path.isdir(os.path.join(dir_, subdir)):
            classes.append(subdir)
    labels = list(chain(*[[i] * image_counts[class_] for i, class_ in enumerate(classes)]))
    nb_labels = len(set(labels))
    labels = to_categorical(labels)
    return labels, nb_labels

def sample_bottleneck_features():
    # Transfer learning
    model = InceptionV3(weights='imagenet', include_top=False)

    # nothing else. no augmention. we have plenty of images
    datagen = ImageDataGenerator(rescale=1./255)

    # Training
    print 'generating training bottlenecks'
    generator = datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(img_width, img_height),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)

    nb_samples_rounded = int(SAMPLE_SIZE - SAMPLE_SIZE % float(BATCH_SIZE))
    x, y = model.bottleneck_generator(generator, nb_samples_rounded, nb_worker = NB_WORKERS)

    np.save(open("saved_models/bottleneck_x_{}.npy".format(todaysdate), 'w'), x)
    np.save(open("saved_models/bottleneck_y_{}.npy".format(todaysdate), 'w'), y)

    print 'generated training bottlenecks'
    print('*'*10)

    # Validation
    print 'generating validation bottlenecks'
    generator = datagen.flow_from_directory(
            VALID_DATA_DIR,
            target_size=(img_width, img_height),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True)
    nb_validation_samples_rounded = int(VAL_SAMPLE_SIZE - VAL_SAMPLE_SIZE % float(BATCH_SIZE))
    x_val, y_val = model.bottleneck_generator(generator, nb_validation_samples_rounded, nb_worker = NB_WORKERS)
    print 'generated validation bottlenecks'
    np.save(open("saved_models/bottleneck_x_val_{}.npy".format(todaysdate), 'w'), x_val)
    np.save(open("saved_models/bottleneck_y_val_{}.npy".format(todaysdate), 'w'), y_val)
    print 'bottleneck features saved....'
    print('*'*10)

def train_top_model(date_=time.strftime("%Y_%m_%d")):

    def _load_bottlenecks(model_date):
            x = np.load(open("saved_models/bottleneck_x_{}.npy".format(todaysdate)))
            y = np.load(open("saved_models/bottleneck_y_{}.npy".format(todaysdate)))
            x_val = np.load(open("saved_models/bottleneck_x_val_{}.npy".format(todaysdate)))
            y_val = np.load(open("saved_models/bottleneck_y_val_{}.npy".format(todaysdate)))
            x, y = x[0,:,:,:], y[0,:,:]
            x_val, y_val = x_val[0,:,:,:], y_val[0,:,:]
            return (x, y, x_val, y_val)

    todaysdate = '2017_01_09'
    x, y, x_val, y_val = _load_bottlenecks(todaysdate)
    print('found saved bottleneck features with shape: {}'.format(x.shape))

    print('*'*10)
    print('training the final layer....')
    model = Sequential()
    model.add(Flatten(input_shape=x.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )

    datagen = ImageDataGenerator()
    generator = datagen.flow(x,y)

    # scans the entire dataset for each epoch
    hist_ = model.fit(x, y,
        batch_size = BATCH_SIZE,
        shuffle=True,
        validation_data = (x_val, y_val),
        nb_epoch = nb_epoch)

    #, validation_split= 0.1, validation_data = (x_val, y_val, val_sample_weights) )
    model.save_weights(TOP_MODEL_WEIGHTS_PATH)
    print 'model trained'
    return hist_

if __name__ == "__main__":
    start = time.time()
    sample_bottleneck_features()
    t1 = time.time()
    print 'bottleneck generation took {} seconds...'.format( t1 - start)
    hist_ = train_top_model()
    t2 = time.time()
    print 'training the model took {} seconds...'.format(t2 - t1)
