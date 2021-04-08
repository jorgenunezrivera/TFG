from check_dataset_env import check_dataset_posibilities
from deep_q_learning import Estimator, make_epsilon_greedy_policy_from_list
from deep_q_learning_validation import validation
from random_env_test import random_env_test, random_env_test_batch
import numpy as np
from window_env_generator import ImageWindowEnvGenerator

TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"
envs=[]
envs.append(ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE, 6, 32,intermediate_rewards=0, continue_until_dies=0, n_actions=3, best_reward=1,
                              no_label_eval=1))
envs.append(ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE, 8, 16,intermediate_rewards=0, continue_until_dies=0, n_actions=3, best_reward=1,
                              no_label_eval=1))
envs.append(ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE, 6, 32,intermediate_rewards=0, continue_until_dies=1, n_actions=4, best_reward=1,
                              no_label_eval=1))
envs.append(ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE, 8, 16,intermediate_rewards=0, continue_until_dies=1, n_actions=4, best_reward=1,
                              no_label_eval=1))

for i,env in enumerate(envs):
    print("env {} max_steps:{} step_size:{} continue_until_dies:{} n_actions:{} best_reward:{}, no_label_eval:{}".format
          (i,env.max_steps,env.step_size,env.continue_until_dies,env.n_actions,env.best_reward,env.no_label_eval))
    random_env_test_batch(env,10)

#