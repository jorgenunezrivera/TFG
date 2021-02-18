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

def validation(q_estimator, env):
    rewards = []
    hits=0
    incorrect_prediction_certainty=0
    for i in range(len(env)):
        obs = env.reset()
        for _ in itertools.count():
            q_values = q_estimator.predict(np.array([obs]))
            best_action = np.argmax(q_values)
            obs, reward, done, info = env.step(best_action)
            if (done):
                if (info["hit"]):
                    hits += 1
                else:
                    incorrect_prediction_certainty+=info["max_prediction_value"]
                rewards.append(reward)
                break
    return np.mean(rewards),hits/len(env),incorrect_prediction_certainty/(len(env)-hits)
