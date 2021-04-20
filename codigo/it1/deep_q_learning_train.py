
import json
from window_env_generator import ImageWindowEnvGenerator
from datetime import datetime
import time
from deep_q_learning import deep_q_learning, Estimator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Env files
TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"
#Constants
REPLAY_MEMORY_SIZE=10000
REPLAY_MEMORY_INIT_SIZE=1000

#training default params
NUM_EPISODES = 12000
LEARNING_RATE = 0.00001
UPDATE_TARGET_ESTIMATOR_EVERY=120
VALIDATE_EVERY=4000
#env default params
#Default parameters
MAX_STEPS = 6
STEP_SIZE = 32
CONTINUE_UNTIL_DIES = 0
MODEL_NAME="atari"


def deep_q_learning_train(num_episodes=NUM_EPISODES,learning_rate=LEARNING_RATE,update_target_freq=UPDATE_TARGET_ESTIMATOR_EVERY,
                          validate_freq=VALIDATE_EVERY,max_steps=MAX_STEPS,step_size=STEP_SIZE,
                 continue_until_dies=CONTINUE_UNTIL_DIES,model_name=MODEL_NAME):

    env = ImageWindowEnvGenerator(TRAINING_IMAGES_DIR, TRAINING_LABELS_FILE,max_steps,step_size,continue_until_dies,best_reward=1,no_label_eval=0)

    validation_env = ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE,max_steps,step_size,continue_until_dies,best_reward=1,no_label_eval=1)

    N_ACTIONS = env.action_space.n
    IMG_SHAPE = env.observation_space.shape

    initial_ts = time.time()

    q_estimator = Estimator(IMG_SHAPE, N_ACTIONS, learning_rate,model_name)
    target_estimator = Estimator(IMG_SHAPE, N_ACTIONS, learning_rate,model_name)
    stats = deep_q_learning(env, q_estimator,
                                                                                             target_estimator,
                                                                                             validation_env,

                                                                                             num_episodes=num_episodes,
                                                                                             replay_memory_size=REPLAY_MEMORY_SIZE,
                                                                                             replay_memory_init_size=REPLAY_MEMORY_INIT_SIZE,
                                                                                             update_target_estimator_every=update_target_freq,
                                                                                             validate_every=validate_freq,
                                                                                             rewards_mean_every=100,
                                                                                             discount_factor=1,
                                                                                             epsilon_start=1,
                                                                                             epsilon_end=0.1,
                                                                                             epsilon_decay_steps=num_episodes * 4,
                                                                                             batch_size=32)

    training_time=stats["total_time"]-stats["validation_time"]
    print("Training time: " + str(training_time))
    print("secs/episode:" + str(training_time / num_episodes))
    num_validations=int(num_episodes/validate_freq)
    val_time=stats["validation_time"]
    print("Validation time: " + str(val_time))
    print("secs/episode:" + str(val_time / (num_validations*len(validation_env))))

    now = datetime.now()
    #print(stats)
    log_filename = now.strftime("logs/%d_%m_%Y_%H_%M_%S_log.json")
    with open(log_filename, 'w') as fp:
        json.dump(stats, fp)
    return log_filename
