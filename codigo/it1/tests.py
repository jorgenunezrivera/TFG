from deep_q_learning import Estimator
from deep_q_learning_validation import validation
from random_env_test import random_env_test
from window_env_generator import ImageWindowEnvGenerator
TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"
env = ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE,18,32,0,0)

env.reset()
for t in range(100000):
    env.random_window()
    print(t)

#random_reward,random_hits=random_env_test(env)


#N_ACTIONS = env.action_space.n
#IMG_SHAPE = env.observation_space.shape
#q_estimator = Estimator(IMG_SHAPE, N_ACTIONS, 0.00001)
#q_estimator.load_model()
#validation(q_estimator,env)
