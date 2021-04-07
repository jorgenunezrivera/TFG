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
import scipy
from scipy import io
import json

from deep_q_learning import deep_q_learning, Estimator
from window_env_generator import ImageWindowEnvGenerator


def sliding_window(index, env,max_zoom):
    env.restart_in_state(index)
    for z in range(max_zoom):
        for y in range(z):
            for x in range(z):
                # print("sample {}  x,y,z  ={},{},{}".format(index, x, y, z))
                if(x+y+z)>env.max_steps:
                    break
                env.set_window(x, y, z)
                _, _, _, info = env.step(3)
                if (info["hit"]):
                    #print("sample {} hit with x, y,z  ={},{},{}, certainty:{}".format(index, x, y, z,
                    #                                                                  info["max_prediction_value"]))
                    return True
    #print("sample {} wrong".format(index))
    return False

def check_dataset_posibilities(env,max_zoom):
    wrongs = []
    hits = 0
    for i in range(len(env)):
        _ = env.reset()
        _, _, _, info = env.step(3)
        hit = info["hit"]
        if not hit:
            wrongs.append(i)
        else:
            hits += 1

    #print("Hits: {}/{} ({}%)".format(hits, len(env), hits * 100 / len(env)))

    fixable_wrongs = []
    for w in wrongs:
        if sliding_window(w, env,max_zoom):
            fixable_wrongs.append(w)

    print(
        " {} hits, {} wrongs, {} fixable wrongs with step size: {} and MAX_STEPS: {}, max precission:{}".format(
            hits, len(wrongs), len(fixable_wrongs), env.step_size, env.max_steps, (hits + len(fixable_wrongs)) * 100 / len(env)))
    return
