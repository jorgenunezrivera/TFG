from deep_q_learning import Estimator, make_epsilon_greedy_policy_from_list
from deep_q_learning_validation import validation
from random_env_test import random_env_test
import numpy as np
from window_env_generator import ImageWindowEnvGenerator
TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"
env = ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE,18,32,0,0)



N_ACTIONS = env.action_space.n
IMG_SHAPE = env.observation_space.shape

q_estimator = Estimator(IMG_SHAPE, N_ACTIONS, 0.00001)
policy = make_epsilon_greedy_policy_from_list(
    q_estimator,
    env.action_space.n)

epsilons = np.linspace(1, 0.1, 10000)


# Populate the replay memory with initial experience

state = env.reset()
# state = np.stack([state] * 4, axis=2)
for i in range(10000):
    legal_actions = env.get_legal_actions()
    if (len(legal_actions) == 0):
        print("ERROR: NO LEGAL ACTIONS POSSIBLE")
        break;
    action_probs = policy(state, epsilons[min(i, 9999)], legal_actions)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    next_state, reward, done, _ = env.step(action)
    #replay_memory.append(Transition(state, action, reward, next_state, done))
    if done:
        state = env.reset()
    else:
        state = next_state

#random_reward,random_hits=random_env_test(env)


#N_ACTIONS = env.action_space.n
#IMG_SHAPE = env.observation_space.shape
#q_estimator = Estimator(IMG_SHAPE, N_ACTIONS, 0.00001)
#q_estimator.load_model()
#validation(q_estimator,env)
