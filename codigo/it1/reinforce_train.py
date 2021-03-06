import json
from window_env_generator import ImageWindowEnvGenerator
from datetime import datetime
import time
from deep_q_learning import deep_q_learning, Estimator
import os
from reinforce import reinforce,ValueEstimator, PolicyEstimator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# IMG_SHAPE=(224,224,3)
TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"

NUM_EPISODES = 12000
LEARNING_RATE = 0.000001
VALIDATE_EVERY=3000




env = ImageWindowEnvGenerator(TRAINING_IMAGES_DIR, TRAINING_LABELS_FILE)
validation_env = ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE)

N_ACTIONS = env.action_space.n
IMG_SHAPE = env.observation_space.shape

initial_ts = time.time()

policy_estimator = PolicyEstimator(IMG_SHAPE, N_ACTIONS, LEARNING_RATE)
value_estimator = ValueEstimator(IMG_SHAPE, LEARNING_RATE)

stats=reinforce(env,policy_estimator,value_estimator,NUM_EPISODES,validation_env,validate_every=VALIDATE_EVERY)

elapsed_time = time.time() - initial_ts
print("Elapsed time: " + str(elapsed_time))
print("Num episodes: " + str(NUM_EPISODES))
print("secs/episode:" + str(elapsed_time / NUM_EPISODES))

now = datetime.now()
print(stats)
log_filename = now.strftime("logs/%d_%m_%Y_%H_%M_%S_reinforce_log.json")
with open(log_filename, 'w') as fp:
    json.dump(stats, fp)

