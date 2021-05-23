import math

import tensorflow as tf
import numpy as np

import itertools

from time import time
import json

from reinforce import PolicyEstimator
from window_env import ImageWindowEnv

with open("label_to_index_dict.json", "r") as read_file:
    label_index_dict = json.load(read_file)

VALIDATION_LABELS_FILE="validation_labels.txt"
VALIDATION_IMAGES_DIR="validation"

FILES_TO_TEST=25

action_estimator=PolicyEstimator((224,224,3),3,0.00001,"atari")
action_estimator.load_model("")
seconds=time()

env=ImageWindowEnv(VALIDATION_IMAGES_DIR,VALIDATION_LABELS_FILE,max_steps=6,step_size=32,continue_until_dies=1,is_validation=1)


for i in range(FILES_TO_TEST):
    print("sample : {} ".format(i))
    best_reward = 0
    obs=env.reset()
    done=False
    best_window=(0,0,0)
    for t in itertools.count():
        action_probs = action_estimator.predict(np.array([obs]))[0]
        action_probs = tf.nn.softmax(action_probs).numpy()
        legal_actions = env.get_legal_actions()
        for i in range(len(action_probs)):
            if i not in legal_actions:
                action_probs[i] = 0
        if np.sum(action_probs) == 0 or math.isnan(sum(action_probs)):
            print("action probs error: sum action_probs =0")
            break;
        action_probs = action_probs / np.sum(action_probs)
        chosen_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        obs, reward, done, info = env.step(chosen_action)
        print("position: {} , reward: {}".format(info["position"],reward))
        if reward>best_reward:
            best_reward=reward
            best_window=(info["position"])
        if(done):
            print("best position : {}  Hit : {}".format(best_window,info["final_hit"]))
            env.set_window(best_window[0],best_window[1],best_window[2])
            env.render()
            break


