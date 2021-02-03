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


VALIDATION_IMAGES_DIR="validation"
q_estimator=Estimator((160,160,3),5)
q_estimator.load_model()
seconds=time()
image_batch=[]

for entry in os.listdir(VALIDATION_IMAGES_DIR):
    filename=os.path.join(VALIDATION_IMAGES_DIR,entry)
    if os.path.isfile(filename) and filename.endswith('.JPEG'):
        image =tf.keras.preprocessing.image.load_img(filename)
        img_arr = keras.preprocessing.image.img_to_array(image)
        image_batch.append(img_arr)
        
env=ImageWindowEnvBatch(image_batch)
load_time=time()-seconds
print("load time: " + str(load_time))

rewards = []

for i in range(25):
    predicted_class=0
    obs=env.reset()
    env.render
    done=False
    for t in itertools.count():
        q_values = q_estimator.predict(np.array([obs]))
        best_action = np.argmax(q_values)
        obs, reward, done, info = env.step(best_action)
        new_predicted_class=info["predicted_class"]
        predicted_class=new_predicted_class
        if(done):
            rewards.append(reward)
            break
validate_time=time()-seconds-load_time
print("validate time: " + str(validate_time))
print("rewards mean:")
print(np.mean(rewards))

plt.figure(figsize=(8, 8))
plt.plot(rewards, label='Rewards')
plt.legend(loc='upper right')
plt.ylabel('Rewards')
plt.ylim([0,1])
plt.title('Rewards')
plt.xlabel('sample')
plt.show()
