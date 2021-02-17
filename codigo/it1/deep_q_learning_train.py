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
import time



from deep_q_learning import deep_q_learning, Estimator

#IMG_SHAPE=(224,224,3)
TRAINING_IMAGES_DIR="train_200"
VALIDATION_IMAGES_DIR="validation"
TRAINING_LABELS_FILE="training_labels.txt"
VALIDATION_LABELS_FILE="validation_labels.txt"
NUM_EPISODES=120


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

N_ACTIONS=env.action_space.n
IMG_SHAPE=env.observation_space.shape

initial_ts=time.time()

q_estimator=Estimator(IMG_SHAPE,N_ACTIONS)
target_estimator=Estimator(IMG_SHAPE,N_ACTIONS)
training_losses, training_rewards, validation_rewards =deep_q_learning(env,q_estimator,target_estimator,validation_env,num_episodes=NUM_EPISODES,replay_memory_size=10000,
                      replay_memory_init_size=2000,update_target_estimator_every=500,validate_every=1000,rewards_mean_every=20,discount_factor=1,
                      epsilon_start=1,epsilon_end=0.1,epsilon_decay_steps=NUM_EPISODES*5, batch_size=32)

elapsed_time=time.time()-initial_ts
print("Elapsed time: " + str(elapsed_time))
print("Num episodes: " + str(NUM_EPISODES))
print("secs/episode:" + str(elapsed_time/NUM_EPISODES))

print("training_losses:{} ".format(training_losses))
print("training_rewards: {}".format(training_rewards))

training_losses_x= [x[0] for x in training_losses]
training_losses_y= [x[1] for x in training_losses]
training_rewards_x=[x[0] for x in training_rewards]
training_rewards_y=[x[1] for x in training_rewards]
validation_rewards_x=[x[0] for x in validation_rewards]
validation_rewards_y=[x[1] for x in validation_rewards]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(training_losses_x,training_losses_y)
plt.legend(loc='upper right')
plt.ylabel('Mean Squared Error')
plt.ylim([0,1])
plt.title('Training Loss')
plt.xlabel('epoch')


plt.subplot(2, 1, 2)
plt.plot(training_rewards_x,training_rewards_y)
plt.plot(validation_rewards_x,validation_rewards_y,'ro')
plt.legend(loc='upper right')
plt.ylabel('Rewards')
plt.ylim([-1.1,1.1])
plt.title('Training Rewards')
plt.xlabel('epoch')


plt.show()


validation_reward_list=[x[1] for x in validation_rewards]
validation_reward_mean=np.mean(validation_reward_list)
validation_reward_variance=np.var(validation_reward_list)
print("validation reward : mean: " + str(validation_reward_mean)+ " variance: " + str(validation_reward_variance))
