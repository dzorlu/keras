#!/usr/bin/python
from __future__ import absolute_import

from ..preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from ..models import Sequential
from ..layers import Convolution2D, MaxPooling2D
from ..layers import Activation, Dropout, Flatten, Dense, Input, InputLayer
from ..applications.inception_v3 import InceptionV3
from ..applications.vgg16 import VGG16
from ..applications.imagenet_utils import preprocess_input, _obtain_input_shape
from .. import activations, initializations, regularizers, constraints
from ..utils.np_utils import to_categorical

import os
import numpy as np
from collections import OrderedDict
from itertools import chain
import time

import boto.s3
from boto.s3.key import Key
import zipfile

# dimensions of our images.
img_width, img_height = 299, 299
todaysdate = time.strftime("%Y_%m_%d")


IMAGE_BUCKET = "item-images-01"
IMAGE_FILE_NAME = "data.zip"
MODEL_BUCKET = "trained-models-keras"
IMAGE_DIRECTORY = "/tmp/"
REGION = "us-east-1"

TRAIN_DATA_DIR = IMAGE_DIRECTORY + "train"
VALID_DATA_DIR = IMAGE_DIRECTORY + "validation"

#TRAIN_DATA_DIR = '/Users/denizzorlu/style/data/train'
#VALID_DATA_DIR = '/Users/denizzorlu/style/data/validation'
TOP_MODEL_WEIGHTS_PATH = IMAGE_DIRECTORY + "bottleneck_fc_model.h5"
nb_epoch = 50
BATCH_SIZE = 64
SAMPLE_SIZE = BATCH_SIZE * 400
VALIDATION_FRACTION = 0.2
VAL_SAMPLE_SIZE = SAMPLE_SIZE * VALIDATION_FRACTION
NB_WORKERS = 4


""""""
#TODO: Move this to util folder

def retrieve_images():
    retrieve_from_s3(incoming_bucket = IMAGE_BUCKET, file_to_retrieve = IMAGE_FILE_NAME)
    unzip_file(IMAGE_FILE_NAME)
    return key_string

def unzip_file(file_to_unzip):
    zip_ref = zipfile.ZipFile(IMAGE_DIRECTORY + file_to_unzip , 'r')
    zip_ref.extractall(IMAGE_DIRECTORY)
    zip_ref.close()

def persist_to_s3(target_bucket, file_to_persist):
    # send it to s3ta
    s3c = boto.s3.connect_to_region(REGION)
    b = s3c.get_bucket(target_bucket)
    k = Key(b)
    k.key = date.today().strftime("%m-%d-%Y")
    k.set_contents_from_filename(file_to_persist)

def persist_model_to_s3(model):
    model.save_weights(TOP_MODEL_WEIGHTS_PATH)
    persist_to_s3(MODEL_BUCKET,TOP_MODEL_WEIGHTS_PATH)

def retrieve_from_s3(incoming_bucket, file_to_retrieve):
    # send it to s3.
    s3c = boto.s3.connect_to_region(REGION)
    b = s3c.get_bucket(incoming_bucket)
    bucket_list = b.list()
    for l in bucket_list:
        keyString = str(l.key)
        if keyString == file_to_retrieve:
            l.get_contents_to_filename(IMAGE_DIRECTORY + keyString)
            print("{} extracted".format(keyString))

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

def get_bottleneck_file_paths(val_):
    val_ = ""
    if validation:
        val_ = "val_"
    file_path_x = "{}bottleneck_x_{}.npy".format(IMAGE_DIRECTORY, val_)
    file_path_y = "{}bottleneck_y_{}.npy".format(IMAGE_DIRECTORY,val_)
    return file_path_x, file_path_y

def save_bottleneck_features(x,y,validation=False):
    filename_x, filename_y = get_bottleneck_filenames(validation)
    np.save(open(filename_x, 'w'), x)
    np.save(open(filename_y, 'w'), y)
    persist_to_s3(MODEL_BUCKET, filename_x)
    persist_to_s3(MODEL_BUCKET, filename_y)


def load_bottleneck_features(validation=False):
    file_path_x, file_path_y = get_bottleneck_file_paths(validation)
    retrieve_from_s3(MODEL_BUCKET, filename_x)
    retrieve_from_s3(MODEL_BUCKET, filename_y)

    x = np.load(open(file_path_x))
    y = np.load(open(file_path_y))

    x, y = x[0,:,:,:], y[0,:,:]
    return (x, y)

""""""

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

    save_bottleneck_features(x,y)
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
    save_bottleneck_features(x_val,y_val, validation=True)


    print 'bottleneck features saved....'
    print('*'*10)

def train_top_model(date_=time.strftime("%Y_%m_%d")):
    x, y = load_bottleneck_features()
    x_val, y_val = load_bottleneck_features(validation=True)

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

    persist_model_to_s3(model)

    print 'model trained'
    return hist_

if __name__ == "__main__":
    key_string = retrieve_images()
    start = time.time()
    sample_bottleneck_features(key_string)
    t1 = time.time()
    print 'bottleneck generation took {} seconds...'.format( t1 - start)
    hist_ = train_top_model()
    t2 = time.time()
    print 'training the model took {} seconds...'.format(t2 - t1)
