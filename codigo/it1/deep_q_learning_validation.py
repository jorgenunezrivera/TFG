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

#from deep_q_learning import Estimator
#TODO:loader que cargue 10000 imagenes del disco duro
#Calcular porcentaje de acciertos
#En los fallos, visaualizar certeza

NUM_VALIDATION_IMAGES = 25 #max 100

def validation(q_estimator,env):
    rewards = []
    for i in range(NUM_VALIDATION_IMAGES):
        predicted_class=0
        obs=env.reset()
        done=False
        for t in itertools.count():
            q_values = q_estimator.predict(np.array([obs]))
            best_action = np.argmax(q_values)
            #if(i==24):
            #    print("validation action: "+ str(best_action))
            #    print("validation q_values: " + str(q_values))
            obs, reward, done, info = env.step(best_action)
            if(done):
                rewards.append(reward)
                break
    return np.mean(rewards)
