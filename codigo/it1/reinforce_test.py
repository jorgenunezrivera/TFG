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

FILES_TO_TEST=5
n_class_better=0
n_class_worse=0
n_class_indif=0
n_class_same=0
show=0
end_loop=0

action_estimator=PolicyEstimator((224,224,3),3,0.00001,"mobilenet")
action_estimator.load_model("mobilenet")
seconds=time()

env=ImageWindowEnv(VALIDATION_IMAGES_DIR,VALIDATION_LABELS_FILE,max_steps=6,step_size=32,continue_until_dies=1,is_validation=1)


for i in range(len(env)):
    if  n_class_same > FILES_TO_TEST and n_class_worse > FILES_TO_TEST and n_class_better > FILES_TO_TEST and n_class_indif > FILES_TO_TEST:
        break
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
        if (done):
            class_change = info["class_change"]
            initial_hit = info["initial_hit"]
            hit = info["final_hit"]
            if class_change:
                if hit:
                    n_class_better += 1
                    if (n_class_better <= FILES_TO_TEST):
                        show = 1
                        print("Cambio de clase incorrecta a clase correcta")
                else:
                    if initial_hit:
                        n_class_worse += 1
                        if (n_class_worse <= FILES_TO_TEST):
                            print("Cambio de clase correcta a clase incorrecta")
                            show = 1
                    else:
                        n_class_indif += 1
                        if (n_class_indif <= FILES_TO_TEST):
                            print("Cambio de clase incorrecta a otra clase incorrecta")
                            show = 1
            else:
                n_class_same += 1
                if n_class_same <= FILES_TO_TEST:
                    print("Sin cambio de clase")
                    show = 1
            if show:
                print("best position : {} hit: {} ".format(best_window, info["final_hit"]))
                env.set_window(best_window[0], best_window[1], best_window[2])
                env.render()
            break


