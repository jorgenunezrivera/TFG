import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import Memory
from rl.policy import EpsGreedyQPolicy
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

N_ACTIONS=6

IMG_SHAPE=(224,224,3)
TRAINING_IMAGES_DIR="train"
VALIDATION_IMAGES_DIR="validation"
TRAINING_LABELS_FILE="training_labels.txt"
VALIDATION_LABELS_FILE="validation_labels.txt"


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
        
#env = gym.make("BreakoutDeterministic-v4")
nb_actions = env.action_space.n

#

model = keras.Sequential([
          layers.Conv2D(32, (8,8),strides=(4,4), padding='same', activation='relu',input_shape=IMG_SHAPE),
          #layers.MaxPooling2D(),
          layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu'),
          #layers.MaxPooling2D(),
          layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
          #layers.MaxPooling2D(),
          layers.Flatten(),  
          layers.Dense(512, activation='relu'),
          layers.Dense(nb_actions, activation='softmax')
        ])
model.summary()

memory = Memory(window_length=1)
policy = EpsGreedyQPolicy(0.05)
rlagent = DQNAgent(model,enable_double_dqn=False,nb_actions=nb_actions,memory=memory,policy=policy)
rlagent.compile(Adam(lr=.05),metrics=['mae'])
rlagent.fit(env,1000,verbose=2)

rlagent.test(env,5)

obs= env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
