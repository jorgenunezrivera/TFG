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

N_ACTIONS=6

from deep_q_learning import deep_q_learning, Estimator
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
      

validation_image_batch=[]
validationlist=os.listdir(VALIDATION_IMAGES_DIR)
validationlist.sort()
for entry in validationlist:
    filename=os.path.join(VALIDATION_IMAGES_DIR,entry)
    if os.path.isfile(filename) and filename.endswith('.JPEG'):
        image =tf.keras.preprocessing.image.load_img(filename)
        img_arr = keras.preprocessing.image.img_to_array(image)
        validation_image_batch.append(img_arr)

validation_labels=[]
with open(VALIDATION_LABELS_FILE) as fp:
   line = fp.readline()
   while line:
       validation_labels.append(int(line))
       line = fp.readline()
            
env=ImageWindowEnvBatch(image_batch,training_labels)
        
validation_env=ImageWindowEnvBatch(validation_image_batch,validation_labels)

q_estimator=Estimator(IMG_SHAPE,N_ACTIONS)
target_estimator=Estimator(IMG_SHAPE,N_ACTIONS)
episode_losses, episode_rewards, validation_rewards =deep_q_learning(env,q_estimator,target_estimator,validation_env,num_episodes=6,replay_memory_size=10000,
                      replay_memory_init_size=64,update_target_estimator_every=18,discount_factor=1,
                      epsilon_start=1,epsilon_end=0.1,epsilon_decay_steps=60000, batch_size=32)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(episode_losses,  label='Training Loss')
plt.legend(loc='upper right')
plt.ylabel('Mean Squared Error')
plt.ylim([0,0.7])
plt.title('Training Loss')
plt.xlabel('epoch')


plt.subplot(2, 1, 2)
plt.plot(episode_rewards, label='Rewards')
plt.plot(validation_rewards,'ro', label='Validation Rewards')
plt.legend(loc='upper right')
plt.ylabel('Rewards')
plt.ylim([-1.1,1.1])
plt.title('Training Rewards')
plt.xlabel('epoch')


plt.show()

training_reward_mean=np.mean(episode_rewards)
training_reward_variance=np.var(episode_rewards)
print("training reward : mean: " + str(training_reward_mean)+ " variance: " + str(training_reward_variance))

validation_reward_mean=np.mean(validation_rewards[:][1])
validation_reward_variance=np.var(validation_rewards[:][1])
print("validation reward : mean: " + str(validation_reward_mean)+ " variance: " + str(validation_reward_variance))
print(validation_rewards)
