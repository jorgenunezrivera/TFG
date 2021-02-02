import gym
from window_env_batch import ImageWindowEnvBatch

import tensorflow as tf
from tensorflow import keras

import numpy as np
from stable_baselines3.common.env_checker import check_env


TEST_FILE="test/IMG_20210201_233826.jpg"

image=tf.keras.preprocessing.image.load_img(TEST_FILE)
img_arr= keras.preprocessing.image.img_to_array(image)

env=ImageWindowEnvBatch(np.array([img_arr]))
check_env(env, warn=True)

obs=env.reset()
env.render()
for i in range(10):
    action=np.random.randint(5)
    obs, rewards, done, info = env.step(action)
    print(info)
    print(rewards)
    env.render()

