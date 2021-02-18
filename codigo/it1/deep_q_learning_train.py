from window_env_generator import ImageWindowEnvGenerator
import numpy as np
import matplotlib.pyplot as plt
import time
from deep_q_learning import deep_q_learning, Estimator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# IMG_SHAPE=(224,224,3)
TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"
NUM_EPISODES = 3600
LEARNING_RATE = 0.000001
UPDATE_TARGET_ESTIMATOR_EVERY=200
VALIDATE_EVERY=200


training_labels = []
with open(TRAINING_LABELS_FILE) as fp:
    line = fp.readline()
    while line:
        training_labels.append(int(line))
        line = fp.readline()

validation_labels = []
with open(VALIDATION_LABELS_FILE) as fp:
    line = fp.readline()
    while line:
        validation_labels.append(int(line))
        line = fp.readline()

env = ImageWindowEnvGenerator(TRAINING_IMAGES_DIR, training_labels)

validation_env = ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, validation_labels)

N_ACTIONS = env.action_space.n
IMG_SHAPE = env.observation_space.shape

initial_ts = time.time()

q_estimator = Estimator(IMG_SHAPE, N_ACTIONS, LEARNING_RATE)
target_estimator = Estimator(IMG_SHAPE, N_ACTIONS, LEARNING_RATE)
training_losses, training_rewards, validation_rewards, validation_hits = deep_q_learning(env, q_estimator,
                                                                                         target_estimator,
                                                                                         validation_env,
                                                                                         num_episodes=NUM_EPISODES,
                                                                                         replay_memory_size=10000,
                                                                                         replay_memory_init_size=2000,
                                                                                         update_target_estimator_every=UPDATE_TARGET_ESTIMATOR_EVERY,
                                                                                         validate_every=VALIDATE_EVERY,
                                                                                         rewards_mean_every=50,
                                                                                         discount_factor=1,
                                                                                         epsilon_start=1,
                                                                                         epsilon_end=0.1,
                                                                                         epsilon_decay_steps=NUM_EPISODES * 4,
                                                                                         batch_size=32)

#################################################### PLOTTING RESULTS ###################################################

elapsed_time = time.time() - initial_ts
print("Elapsed time: " + str(elapsed_time))
print("Num episodes: " + str(NUM_EPISODES))
print("secs/episode:" + str(elapsed_time / NUM_EPISODES))


training_losses_x = [x[0] for x in training_losses]
training_losses_y = [x[1] for x in training_losses]
training_rewards_x = [x[0] for x in training_rewards]
training_rewards_y = [x[1] for x in training_rewards]
validation_rewards_x = [x[0] for x in validation_rewards]
validation_rewards_y = [x[1] for x in validation_rewards]
validation_hits_x = [x[0] for x in validation_hits]
validation_hits_y = [x[1] for x in validation_hits]

plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(training_losses_x, training_losses_y)
plt.legend(loc='upper right')
plt.ylabel('Mean Absolute Error')
plt.ylim([0, 1])
plt.title('Training Loss')
plt.xlabel('epoch')

plt.subplot(3, 1, 2)
plt.plot(training_rewards_x, training_rewards_y)
plt.plot(validation_rewards_x, validation_rewards_y, 'ro')
plt.hlines(0,0,NUM_EPISODES)
plt.legend(loc='upper right')
plt.ylabel('Rewards')
plt.ylim([-1.1, 1.1])
plt.title('Training Rewards')
plt.xlabel('epoch')

plt.subplot(3, 1, 3)
plt.plot(validation_hits_x, validation_hits_y)
plt.hlines(0.73,0,NUM_EPISODES)
plt.legend(loc='upper right')
plt.ylabel('Hits')
plt.ylim([0.6, 0.9])
plt.title('Validation hits')
plt.xlabel('epoch')

plt.show()

validation_reward_list = [x[1] for x in validation_rewards]
validation_reward_mean = np.mean(validation_reward_list)
validation_reward_variance = np.var(validation_reward_list)
print("validation reward : mean: " + str(validation_reward_mean) + " variance: " + str(validation_reward_variance))
