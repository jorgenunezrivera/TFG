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

def dqn_test(model_name):
    with open("label_to_index_dict.json", "r") as read_file:
        label_index_dict = json.load(read_file)

    VALIDATION_LABELS_FILE="validation_labels.txt"
    VALIDATION_IMAGES_DIR="validation1000"

    FILES_TO_TEST=4
    n_class_better=0
    n_class_worse=0
    n_class_indif=0
    n_class_same_positive=0
    n_class_same_negative=0
    hits=0
    show=0

    q_estimator=Q_Estimator((224,224,3),3,0.00001,"mobilenet")
    q_estimator.load_model("mobilenet")
    seconds=time()

    env=ImageWindowEnv(VALIDATION_IMAGES_DIR,VALIDATION_LABELS_FILE,max_steps=6,step_size=32,continue_until_dies=1,is_validation=1)

    print (len(env))
    for i in range(len(env)):
        show=0
        if  n_class_same_positive > FILES_TO_TEST and n_class_same_negative > FILES_TO_TEST and n_class_worse > FILES_TO_TEST \
                and n_class_better > FILES_TO_TEST :
            break
        print("sample : {} ".format(i))
        best_reward = 0
        obs=env.reset()
        done=False
        best_window=(0,0,0)
        for t in itertools.count():
            q_values = q_estimator.predict(tf.expand_dims(obs, axis=0))[0]  # NUMPY?
            legal_actions = env.get_legal_actions()
            best_actions = np.argsort(-q_values)
            for action in best_actions:
                if action in legal_actions:
                    best_action = action
                    break
            obs, reward, done, info = env.step(best_action)
            #print("position: {} , reward: {}".format(info["position"],reward))
            if reward>best_reward:
                best_reward=reward
                best_window=(info["position"])
            if(done):
                class_change = info["class_change"]
                initial_hit = info["initial_hit"]
                hit = info["final_hit"]
                if hit:
                    hits+=1
                if class_change:
                    if hit:
                        n_class_better += 1
                        if(n_class_better<=FILES_TO_TEST):
                            #show=1
                            print("Cambio de clase incorrecta a clase correcta")
                    else:
                        if initial_hit:
                            n_class_worse += 1
                            if(n_class_worse<=FILES_TO_TEST):
                                print("Cambio de clase correcta a clase incorrecta")
                                print(env.get_predicted_class_name())
                                show=1
                        else:
                            n_class_indif += 1
                            if (n_class_indif <= FILES_TO_TEST):
                                print("Cambio de clase incorrecta a otra clase incorrecta")
                           #     show = 1
                else:
                    if hit:
                        n_class_same_positive+=1
                        if(n_class_same_positive<= FILES_TO_TEST):
                            print("Sin cambio de clase,acierto")
                            show=1
                    else:
                        n_class_same_negative += 1
                        if (n_class_same_negative <= FILES_TO_TEST):
                            print("Sin cambio de clase,fallo")
                            print("clase correcta: {}".format(env.get_true_class_name()))
                            print("clase predicha: {}".format(env.get_predicted_class_name()))
                            show = 1
                        #    show = 1
                if(show):
                    print("best position : {} hit: {} ".format(best_window,info["final_hit"]))
                    env.set_window(best_window[0],best_window[1],best_window[2])
                    env.render_with_info()
                break

    print("Same class positive: {} Same class negative {} Class better: {} Class worse: {} Class indiferent: {} Final hits: {}".format(
    n_class_same_positive,
    n_class_same_negative,
    n_class_better,
    n_class_worse,
    n_class_indif,
        hits))

if len(sys.argv)!=2:
    print("Uso :  python deep_q_learning_test.py model_name")
else:
    model_name= sys.argv[1]
    dqn_test(model_name)
