import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from window_env_batch import ImageWindowEnvBatch
import tensorflow as tf
from window_env_batch import ImageWindowEnvBatch
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import itertools
import sys
import os
import random
import matplotlib.pyplot as plt
from time import time
from deep_q_learning import Estimator
import json
NUM_ACTIONS=4
TRAINING_IMAGES_DIR="train_200"
VALIDATION_IMAGES_DIR="validation"
TRAINING_LABELS_FILE="training_labels.txt"

with open("label_to_index_dict.json", "r") as read_file:
    label_index_dict = json.load(read_file)

image_batch=[]

image_batch=[]
filelist=os.listdir(TRAINING_IMAGES_DIR)
filelist.sort()
for entry in filelist:
    filename=os.path.join(TRAINING_IMAGES_DIR,entry)
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

env=ImageWindowEnvBatch(image_batch,training_labels)


rewards = []
hits=0
for i in range(len(image_batch)):
    predicted_class=0
    obs=env.reset()
    env.render
    done=False
    print("sample: " + str(i))
    for t in itertools.count():
        best_action = np.random.randint(NUM_ACTIONS)
        print("Action: "+ str(best_action))
        obs, reward, done, info = env.step(best_action)
        print("Reward: "+ str(reward))
        if(done):
            rewards.append(reward)
            predicted_class=info["predicted_class"]
            true_class= label_index_dict[str(training_labels[i])]
            if(true_class==predicted_class):
                hits+=1
            break

training_reward_mean=np.mean(rewards)
training_reward_variance=np.var(rewards)
print("training reward : mean: " + str(training_reward_mean)+ " variance: " + str(training_reward_variance))
print("Hits: {}/{} ({}%)".format(hits,len(image_batch),100*hits/len(image_batch)))
