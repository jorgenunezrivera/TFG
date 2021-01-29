import gym
from image_window_env import ImageWindowEnv

import tensorflow as tf
from tensorflow import keras

import numpy as np
from stable_baselines3.common.env_checker import check_env


TEST_FILE="test/IMG_20210123_162708.jpg"

image=tf.keras.preprocessing.image.load_img(TEST_FILE)
img_arr= keras.preprocessing.image.img_to_array(image)

env=ImageWindowEnv(img_arr)
check_env(env, warn=True)

obs=env.reset()
env.render()
for i in range(10):
    action=np.random.randint(5)
    obs, rewards, done, info = env.step(action)
    print(rewards)
    print(obs.shape)
    env.render()

