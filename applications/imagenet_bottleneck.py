#!/usr/bin/python
from __future__ import absolute_import

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, InputLayer
from keras.applications.inception_v3 import InceptionV3
from keras import activations, initializations, regularizers, constraints

import os
import numpy as np
from collections import OrderedDict
from itertools import chain
import time
import math
import boto
import os
import sys
from filechunkio import FileChunkIO

# dimensions of our images.
img_width, img_height = 299, 299
todaysdate = time.strftime("%Y_%m_%d")


IMAGE_BUCKET = "item-images-01"
IMAGE_FILE_NAME = "data.zip"
MODEL_BUCKET = "trained-models-keras"
IMAGE_DIRECTORY = "/tmp/"
REGION = "us-east-1"

TRAIN_DATA_DIR =  "data/train"
VALID_DATA_DIR =  "data/validation"


#TRAIN_DATA_DIR = '/Users/denizzorlu/style/data/train'
#VALID_DATA_DIR = '/Users/denizzorlu/style/data/validation'
TOP_MODEL_WEIGHTS_PATH = IMAGE_DIRECTORY + "bottleneck_fc_model.h5"
nb_epoch = 50
BATCH_SIZE = 64
SAMPLE_SIZE = 120000
VAL_SAMPLE_SIZE = 30000
NB_WORKERS = 4


""""""
#TODO: Move this to util folder
def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError

def retrieve_images():
    print("retrieving images..")
    retrieve_from_s3(incoming_bucket = IMAGE_BUCKET, file_to_retrieve = IMAGE_FILE_NAME)
    unzip_file()
    print("images unzipped..")

def unzip_file():
    cmd = 'jar -xf /tmp/data.zip'
    os.system(cmd)

def persist_to_s3(target_bucket, file_to_persist):
    # send it to s3ta
    s3c = boto.connect_s3()
    b = s3c.get_bucket(target_bucket)
    k = Key(b)
    k.key = file_to_persist
    k.set_contents_from_filename(file_to_persist)

def persist_to_s3_multipart(target_bucket, file_to_persist):
    s3c = boto.connect_s3()
    b = s3c.get_bucket(target_bucket)
    source_size = os.stat(file_to_persist).st_size

    # Create a multipart upload request
    mp = b.initiate_multipart_upload(os.path.basename(file_to_persist))

    # Use a chunk size of 500 MiB (feel free to change this)
    chunk_size = 524288000
    chunk_count = int(math.ceil(source_size / float(chunk_size)))

    # Send the file parts, using FileChunkIO to create a file-like object
    # that points to a certain byte range within the original file. We
    # set bytes to never exceed the original file size.
    for i in range(chunk_count):
        offset = chunk_size * i
        bytes = min(chunk_size, source_size - offset)
        with FileChunkIO(file_to_persist, 'r', offset=offset,
                             bytes=bytes) as fp:
                             mp.upload_part_from_file(fp, part_num=i + 1)

    # Finish the upload
    mp.complete_upload()

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

def get_bottleneck_file_paths(validation):
    val_ = ""
    if validation:
        val_ = "_val"
    file_path_x = "{}bottleneck_x{}.npz".format(IMAGE_DIRECTORY, val_)
    file_path_y = "{}bottleneck_y{}.npz".format(IMAGE_DIRECTORY,val_)
    return file_path_x, file_path_y

def persist_to_disk(filename, dat):
    np.savez_compressed(filename, dat)

def save_bottleneck_features(x,y,validation=False):
    file_path_x, file_path_y = get_bottleneck_file_paths(validation)
    persist_to_disk(file_path_x, x)
    persist_to_disk(file_path_y, y)
    persist_to_s3_multipart(MODEL_BUCKET, file_path_x)
    persist_to_s3_multipart(MODEL_BUCKET, file_path_y)


def load_bottleneck_features(validation=False, from_s3=True):
    file_path_x, file_path_y = get_bottleneck_file_paths(validation)
    if from_s3:
        retrieve_from_s3(MODEL_BUCKET, file_path_x)
        retrieve_from_s3(MODEL_BUCKET, file_path_y)
    x = np.load(open(file_path_x))['arr_0']
    y = np.load(open(file_path_y))['arr_0']
    return (x, y)

""""""

def sample_bottleneck_features():
    # Transfer learning
    model = InceptionV3(weights='imagenet', include_top=False)

    # nothing else. no augmention. we have plenty of images
    datagen = ImageDataGenerator(rescale=1./255)

    # Training.
    print 'generating training bottlenecks'
    generator = datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(img_width, img_height),
        batch_size=BATCH_SIZE,
        shuffle=False, #generating bottlenecks sequentially
        )

    nb_samples_rounded = int(SAMPLE_SIZE - SAMPLE_SIZE % float(BATCH_SIZE))
    x, y = model.bottleneck_generator(generator, nb_samples_rounded, nb_worker = NB_WORKERS)
    x, y = x[0].astype('float32'), y[0].astype('float32')

    print('*'*10)
    print 'saving bottleneck features to disk..'
    save_bottleneck_features(x, y)
    print 'generated training bottlenecks..'
    print('*'*10)

    # Validation
    print 'generating validation bottlenecks'
    generator = datagen.flow_from_directory(
            VALID_DATA_DIR,
            target_size=(img_width, img_height),
            batch_size=BATCH_SIZE,
            shuffle=False, #generating bottlenecks sequentially
            )
    nb_validation_samples_rounded = int(VAL_SAMPLE_SIZE - VAL_SAMPLE_SIZE % float(BATCH_SIZE))
    x_val, y_val = model.bottleneck_generator(generator, nb_validation_samples_rounded, nb_worker = NB_WORKERS)
    x_val, y_val = x_val[0].astype('float32'), y_val[0].astype('float32')

    print('*'*10)
    print 'saving bottleneck features to disk..'
    save_bottleneck_features(x_val,y_val, validation=True)
    print 'generated validation bottlenecks..'
    print('*'*10)

    print 'bottleneck features saved..'
    sys.stdout.flush()

def train_top_model():
    x, y = load_bottleneck_features()
    # remove y's that we have not retrieved yet. 
    y = y[:,y.sum(axis=0)>0]

    x_val, y_val = load_bottleneck_features(validation=True)
    y_val = y_val[:,y_val.sum(axis=0)>0]

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

    # datagen = ImageDataGenerator()
    # generator = datagen.flow(x,y, shuffle=True)

    # scans the entire dataset for each epoch
    hist_ = model.fit(x, y,
        batch_size = BATCH_SIZE,
        shuffle=True,
        #validation_data = (x_val, y_val),
        nb_epoch = nb_epoch)

    persist_model_to_s3(model)

    print 'model trained'
    return hist_

if __name__ == "__main__":
    start = time.time()
    args = sys.argv[1:]
    if str_to_bool(args[0]):
        retrieve_images()
        t0 = time.time()
        print 'retrieving images took {} seconds...'.format( t0 - start)
    if str_to_bool(args[1]):
        sample_bottleneck_features()
        t1 = time.time()
        print 'bottleneck generation took {} seconds...'.format( t1 - t0)
    hist_ = train_top_model()
    t2 = time.time()
    print 'training the model took {} seconds...'.format(t2 - t1)
