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
import json

# from deep_q_learning import Estimator
# TODO:loader que cargue 10000 imagenes del disco duro
# Calcular porcentaje de acciertos
# En los fallos, visaualizar certeza

NUM_VALIDATION_IMAGES = 25  # max 100
with open("label_to_index_dict.json", "r") as read_file:
    label_index_dict = json.load(read_file)
VALIDATION_LABELS_FILE = "validation_labels.txt"
validation_labels = []
validation_true_classes = []
with open(VALIDATION_LABELS_FILE) as fp:
    line = fp.readline()
    while line:
        validation_labels.append(int(line))
        validation_true_classes.append(label_index_dict[line.rstrip()])
        line = fp.readline()


def validation(q_estimator, env):
    rewards = []
    hits=0
    incorrect_prediction_certainty=0
    for i in range(NUM_VALIDATION_IMAGES):
        obs = env.reset()
        for t in itertools.count():
            q_values = q_estimator.predict(np.array([obs]))
            best_action = np.argmax(q_values)
            obs, reward, done, info = env.step(best_action)
            if (done):
                if (validation_true_classes[i] == info["predicted_class"]):
                    hits += 1
                else:
                    incorrect_prediction_certainty+=info["max_prediction_value"]
                rewards.append(reward)
                break
    return np.mean(rewards),hits/NUM_VALIDATION_IMAGES,incorrect_prediction_certainty/NUM_VALIDATION_IMAGES
