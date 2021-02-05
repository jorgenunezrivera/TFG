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
IMG_SHAPE=(128,128,3)
IMAGES_DIR="train"
VALIDATION_IMAGES_DIR="validation"

image_batch=[]
for entry in os.listdir(IMAGES_DIR):
    filename=os.path.join(IMAGES_DIR,entry)
    if os.path.isfile(filename) and filename.endswith('.JPEG'):
        image =tf.keras.preprocessing.image.load_img(filename)
        img_arr = keras.preprocessing.image.img_to_array(image)
        image_batch.append(img_arr)

validation_image_batch=[]

for entry in os.listdir(VALIDATION_IMAGES_DIR):
    filename=os.path.join(VALIDATION_IMAGES_DIR,entry)
    if os.path.isfile(filename) and filename.endswith('.JPEG'):
        image =tf.keras.preprocessing.image.load_img(filename)
        img_arr = keras.preprocessing.image.img_to_array(image)
        validation_image_batch.append(img_arr)
            
env=ImageWindowEnvBatch(image_batch)
        
validation_env=ImageWindowEnvBatch(validation_image_batch)

q_estimator=Estimator(IMG_SHAPE,N_ACTIONS)
target_estimator=Estimator(IMG_SHAPE,N_ACTIONS)
episode_losses, episode_rewards, validation_rewards =deep_q_learning(env,q_estimator,target_estimator,validation_env,num_episodes=6000,replay_memory_size=10000,
                      replay_memory_init_size=64,update_target_estimator_every=600,discount_factor=1,
                      epsilon_start=1,epsilon_end=0.1,epsilon_decay_steps=25000, batch_size=32)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(episode_losses, label='Training Loss')
plt.legend(loc='upper right')
plt.ylabel('Mean Squared Error')
plt.ylim([0,0.03])
plt.title('Training Loss')
plt.xlabel('epoch')


plt.subplot(2, 1, 2)
plt.plot(episode_rewards, label='Rewards')
plt.plot(validation_rewards, frmt='ro' label='Validation Rewards')
plt.legend(loc='upper right')
plt.ylabel('Rewards')
plt.ylim([-0.75,0.75])
plt.title('Training Rewards')
plt.xlabel('epoch')


plt.show()


