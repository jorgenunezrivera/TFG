import math
import sys

import tensorflow as tf
import numpy as np

import itertools

from time import time
import json

from reinforce import PolicyEstimator
from window_env import ImageWindowEnv

def reinforce_test_top5(model_name):


    VALIDATION_LABELS_FILE="validation_labels.txt"
    VALIDATION_IMAGES_DIR="validation1000"


    initial_hits=0
    final_hits=0
    initial_top5_hits=0
    final_top5_hits=0

    action_estimator=PolicyEstimator((224,224,3),3,0.00001,model_name)
    action_estimator.load_model(model_name)
    seconds=time()

    env=ImageWindowEnv(VALIDATION_IMAGES_DIR,VALIDATION_LABELS_FILE,max_steps=6,step_size=32,continue_until_dies=1,is_validation=1)


    for i in range(len(env)):

        print("sample : {} ".format(i))
        best_reward = 0
        obs=env.reset()
        done=False
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
                hit = info["final_hit"]
                top5hit = info["final_top5"]
                initial_hit = info["initial_hit"]
                initial_top5 = info["initial_top5"]
                if hit:
                    final_hits += 1
                if top5hit:
                    final_top5_hits += 1
                if initial_hit:
                    initial_hits += 1
                if initial_top5:
                    initial_top5_hits += 1
                break
    print("Env len : {} initial top1 accuracy: {} initial top5 accuracy: {}  final top1 accuracy {} ; top5 accuracy: {}".format(
    len(env),initial_hits/len(env),initial_top5_hits/len(env), final_hits/len(env),final_top5_hits/len(env)))

if len(sys.argv)!=2:
    print("Uso :  python reinforce_test_top5.py model_name")
else:
    model_name= sys.argv[1]
    reinforce_test_top5(model_name)
