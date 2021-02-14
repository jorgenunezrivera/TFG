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

N_ACTIONS=6



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

metadata=scipy.io.loadmat("meta.mat")
dictionary={}
for i in range(1000):
    dictionary[metadata['synsets'][i][0][1][0]]=int(metadata['synsets'][i][0][0][0][0])

with open('label_dict.json', 'w') as fp:
    json.dump(dictionary, fp)

inverse_dictionary={}
for i in range(1000):
    inverse_dictionary[int(metadata['synsets'][i][0][0][0][0])]=metadata['synsets'][i][0][1][0]

with open('inverse_label_dict.json', 'w') as fp:
    json.dump(inverse_dictionary, fp)

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

indexes=[]
index_dict={}
for i in range(1000):
    fake_predictions=np.zeros(1000)
    fake_predictions[i]=1
    synset=tf.keras.applications.mobilenet_v2.decode_predictions(np.array([fake_predictions]), top=1)[0][0][0]
    label=dictionary[synset]
    index_dict[label]=i

print(indexes)
with open('label_to_index_dict.json', 'w') as fp:
    json.dump(index_dict, fp)
rewards=[]
hits=0
for i in range(len(image_batch)):
    image_window_resized=tf.image.resize(image_batch[i],size=(HEIGHT, WIDTH)).numpy()
    image_window_resized=tf.keras.applications.mobilenet_v2.preprocess_input(image_window_resized)
    image_window_expanded=np.array([image_window_resized])
    predictions=model.predict(image_window_expanded)
    label=training_labels[i]
    print("label: " + str(label))
    decoded_predictions=tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    predicted_label=dictionary[decoded_predictions[0][0][0]]
    print("predicted label: " + str(predicted_label))
    reward=decoded_predictions[0][0][2]
    rewards.append(reward)
    if(label==predicted_label):
        hits+=1

print("Hits: {}/{} ({}%)".format(hits,len(image_batch),hits*100/len(image_batch)))
training_reward_mean=np.mean(rewards)
training_reward_variance=np.var(rewards)
print("training reward : mean: " + str(training_reward_mean)+ " variance: " + str(training_reward_variance))
