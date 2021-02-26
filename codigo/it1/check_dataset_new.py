import tensorflow as tf
from window_env_batch import ImageWindowEnvBatch
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple
import itertools
import sys
import os
import random
import matplotlib.pyplot as plt
import scipy
from scipy import io
import json

from deep_q_learning import deep_q_learning, Estimator


HEIGHT=224
WIDTH=224
IMG_SHAPE=(224,224,3)
TRAINING_IMAGES_DIR="train_200"
VALIDATION_IMAGES_DIR="validation"
TRAINING_LABELS_FILE="training_labels.txt"
VALIDATION_LABELS_FILE="validation_labels.txt"

model=model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=True,
                                               weights='imagenet')
with open("label_to_index_dict.json", "r") as read_file:
    label_index_dict = json.load(read_file)

image_batch=[]
filelist=os.listdir(TRAINING_IMAGES_DIR)
filelist.sort()
for entry in filelist:
    filename=os.path.join(TRAINING_IMAGES_DIR,entry)
    print("filename: " + filename)
    if os.path.isfile(filename) and filename.endswith('.JPEG'):
        image =tf.keras.preprocessing.image.load_img(filename)
        img_arr = keras.preprocessing.image.img_to_array(image)
        image_batch.append(img_arr)

training_labels=[]
with open(TRAINING_LABELS_FILE) as fp:
   line = fp.readline()
   while line:
       training_labels.append(int(line))
       line = fp.readline()

hits=0
for i in range(len(image_batch)):
    image_window_resized=tf.image.resize(image_batch[i],size=(HEIGHT, WIDTH)).numpy()
    image_window_resized=tf.keras.applications.mobilenet_v2.preprocess_input(image_window_resized)
    image_window_expanded=np.array([image_window_resized])
    predictions=model.predict(image_window_expanded)
    true_prediction=label_index_dict[str(training_labels[i])]
    prediction=np.argmax(predictions)
    hit=(true_prediction==prediction)
    print("Sample {} true_prediction: {} prediction :{} Hit: {}".format(i,true_prediction,prediction,hit))
    if hit:
        hits+=1

print("Hits: {}/{} ({}%)".format(hits,len(image_batch),hits*100/len(image_batch)))
