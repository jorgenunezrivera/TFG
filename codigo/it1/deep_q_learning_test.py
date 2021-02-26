import tensorflow as tf
from window_env_generator import ImageWindowEnvGenerator
import numpy as np
from tensorflow import keras
import itertools
import os
import matplotlib.pyplot as plt
from time import time
from deep_q_learning import Estimator
import json

with open("label_to_index_dict.json", "r") as read_file:
    label_index_dict = json.load(read_file)

VALIDATION_LABELS_FILE="validation_labels.txt"
VALIDATION_IMAGES_DIR="validation"

FILES_TO_TEST=5

q_estimator=Estimator((224,224,3),6,0.00001)
q_estimator.load_model()
seconds=time()



env=ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR,VALIDATION_LABELS_FILE)
load_time=time()-seconds
print("load time: " + str(load_time))

rewards = []
hits=0
for i in range(FILES_TO_TEST):
    predicted_class=0
    obs=env.reset()

    done=False
    print("sample: " + str(i))
    for t in itertools.count():
        q_values = q_estimator.predict(np.array([obs]))
        best_action = np.argmax(q_values)
        print("q_values : " + str(q_values))
        print("Action: "+ str(best_action))
        obs, reward, done, info = env.step(best_action)
        env.render()
        if(done):
            rewards.append(reward)
            hits+=info["hit"]
            print("reward: "+ str(reward))
            print("hit: "+str(info["hit"]))
            print("top5: "+str(info["top5"]))
            break

validate_time=time()-seconds-load_time
print("validate time: " + str(validate_time))
print("rewards mean:")
print(np.mean(rewards))
correct_predictions=0

print("Correct predictions. {} / {} ({}%)".format(hits,len(env),100*hits/len(env)))


plt.figure(figsize=(8, 8))
plt.plot(rewards, label='Rewards')
plt.legend(loc='upper right')
plt.ylabel('Rewards')
plt.ylim([-1, 1])
plt.title('Rewards')
plt.xlabel('sample')
plt.show()
