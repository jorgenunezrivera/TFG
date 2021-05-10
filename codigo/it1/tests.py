
import numpy as np

from enviroment_tests import random_env_test_batch, check_dataset_posibilities
from window_env import ImageWindowEnv

TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"

env=ImageWindowEnv(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE, 6, 32,continue_until_dies=1, is_validation=1)


check_dataset_posibilities(env, 5)

print("max_steps:{} step_size:{} continue_until_dies:{} n_actions:{} ".format
          (env.max_steps,env.step_size,env.continue_until_dies,env.n_actions))
random_env_test_batch(env,10)

#