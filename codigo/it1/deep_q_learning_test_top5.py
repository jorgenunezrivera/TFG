import sys

import tensorflow as tf
import numpy as np
from tensorflow import keras
import itertools
import os
import matplotlib.pyplot as plt
from time import time
from deep_q_learning import Q_Estimator
import json

from window_env import ImageWindowEnv




def top5test(model_name):

    with open("label_to_index_dict.json", "r") as read_file:
        label_index_dict = json.load(read_file)

    VALIDATION_LABELS_FILE="validation_labels.txt"
    VALIDATION_IMAGES_DIR="validation1000"

    hits=0
    top5hits=0
    initial_hits=0
    initial_top5_hits=0
    q_estimator=Q_Estimator((224,224,3),3,0.00001,model_name)
    q_estimator.load_model(model_name)
    seconds=time()

    env=ImageWindowEnv(VALIDATION_IMAGES_DIR,VALIDATION_LABELS_FILE,max_steps=6,step_size=32,continue_until_dies=1,is_validation=1)

    print (len(env))
    for i in range(len(env)):
        print("sample : {} ".format(i))

        obs=env.reset()
        done=False

        for t in itertools.count():
            q_values = q_estimator.predict(tf.expand_dims(obs, axis=0))[0]  # NUMPY?
            legal_actions = env.get_legal_actions()
            best_actions = np.argsort(-q_values)
            for action in best_actions:
                if action in legal_actions:
                    best_action = action
                    break
            obs, reward, done, info = env.step(best_action)


            if(done):
                hit = info["final_hit"]
                top5hit = info["final_top5"]
                initial_hit=info["initial_hit"]
                initial_top5=info["initial_top5"]
                if hit:
                    hits+=1
                if top5hit:
                    top5hits+=1
                if initial_hit:
                    initial_hits+=1
                if initial_top5:
                    initial_top5_hits+=1
                break

    print("Env len : {} initial top1 accuracy: {} initial top5 accuracy: {}  final top1 accuracy {} ; top5 accuracy: {}".format(
    len(env),initial_hits/len(env),initial_top5_hits/len(env), hits/len(env),top5hits/len(env)))

if len(sys.argv)!=2:
    print("Uso :  python deep_q_learning_test_top5.py model_name")
else:
    model_name= sys.argv[1]
    top5test(model_name)