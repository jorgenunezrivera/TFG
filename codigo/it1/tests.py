from check_dataset_env import check_dataset_posibilities
from deep_q_learning import Estimator, make_epsilon_greedy_policy_from_list
from deep_q_learning_validation import validation
from random_env_test import random_env_test
import numpy as np
from window_env_generator import ImageWindowEnvGenerator
TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"
env = ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE,6,16,0,0,n_actions=3,best_reward=1)
check_dataset_posibilities(env,3)

cumulated_random_reward=0
cumulated_random_hits=0
for i in range(10):
    random_reward,random_hits=random_env_test(env)
    print("Random test. reward: {} hits: {}".format(random_reward,random_hits))
    cumulated_random_reward+=random_reward
    cumulated_random_hits+=random_hits
cumulated_random_hits/=10
cumulated_random_reward/=10
print("10 random tests. mena reward: {} mean hits: {}".format(cumulated_random_reward, cumulated_random_hits))

#N_ACTIONS = env.action_space.n
#IMG_SHAPE = env.observation_space.shape
#q_estimator = Estimator(IMG_SHAPE, N_ACTIONS, 0.00001)
#q_estimator.load_model()
#validation(q_estimator,env)
