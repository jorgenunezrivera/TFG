import gc
import json
import time

import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import namedtuple
import itertools
import random
from datetime import datetime
import os

from build_models import build_dqn_model
from window_env import ImageWindowEnv

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
huber = tf.keras.losses.Huber()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Env files
TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"
# Constants
REPLAY_MEMORY_SIZE = 10000
REPLAY_MEMORY_INIT_SIZE = 1000

# training default params
NUM_EPISODES = 12000
LEARNING_RATE = 0.00001
UPDATE_TARGET_ESTIMATOR_EVERY = 120
VALIDATE_EVERY = 4000
MODEL_NAME = "atari"

# env default params
# Default parameters
MAX_STEPS = 6
STEP_SIZE = 32
CONTINUE_UNTIL_DIES = 0



def deep_q_learning_train(num_episodes=NUM_EPISODES, learning_rate=LEARNING_RATE,
                          update_target_freq=UPDATE_TARGET_ESTIMATOR_EVERY,
                          validate_freq=VALIDATE_EVERY, max_steps=MAX_STEPS, step_size=STEP_SIZE,
                          continue_until_dies=CONTINUE_UNTIL_DIES, model_name=MODEL_NAME):

    env = ImageWindowEnv(TRAINING_IMAGES_DIR, TRAINING_LABELS_FILE, max_steps, step_size, continue_until_dies,
                         is_validation=0)

    validation_env = ImageWindowEnv(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE, max_steps, step_size,
                                    continue_until_dies, is_validation=1)

    n_actions = env.action_space.n
    img_shape = env.observation_space.shape

    q_estimator = Estimator(img_shape, n_actions, learning_rate, model_name)
    target_estimator = Estimator(img_shape, n_actions, learning_rate, model_name)
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

    training_time = stats["total_time"] - stats["validation_time"]
    print("Training time: " + str(training_time))
    print("secs/episode:" + str(training_time / num_episodes))
    num_validations = int(num_episodes / validate_freq)
    val_time = stats["validation_time"]
    print("Validation time: " + str(val_time))
    print("secs/episode:" + str(val_time / (num_validations * len(validation_env))))

    now = datetime.now()
    # print(stats)
    log_filename = now.strftime("logs/%d_%m_%Y_%H_%M_%S_log.json")
    with open(log_filename, 'w') as fp:
        json.dump(stats, fp)
    return log_filename


def validation(q_estimator, env):
    # init_ts=time.time()
    rewards = []
    hits = 0
    class_changes = 0
    class_changes_bad = 0
    class_changes_good = 0
    class_changes_equal = 0
    positive_rewards = 0
    for i in range(len(env)):
        obs = env.reset()
        for _ in itertools.count():
            q_values = q_estimator.predict(tf.expand_dims(obs, axis=0))[0]  # NUMPY?
            legal_actions = env.get_legal_actions()
            best_actions = np.argsort(-q_values)
            for action in best_actions:
                if action in legal_actions:
                    best_action = action
                    break
            obs, reward, done, info = env.step(best_action)
            # if(i%20==0):
            #    print("q values: {}, reward: {} , hit:{}".format(q_values,reward,info["hit"]))
            if done:
                class_change = info["class_change"]
                initial_hit = info["initial_hit"]
                hit = info["final_hit"]
                rewards.append(info["best_reward"])
                if hit:
                    hits += 1
                if class_change:
                    class_changes += 1
                    if hit:
                        class_changes_good += 1
                    else:
                        if initial_hit:
                            class_changes_bad += 1
                        else:
                            class_changes_equal += 1
                if info["best_reward"] > 0:
                    positive_rewards += 1
                break
    # print("time_elapsed={}".format(time.time()-init_ts))
    return np.mean(rewards), hits / len(env), class_changes / len(env), class_changes_good / len(
        env), class_changes_bad / len(env), class_changes_equal / len(env), positive_rewards / len(env)


class Estimator:
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, input_shape, n_actions, learning_rate, model_name):
        self._build_model(input_shape, n_actions, learning_rate, model_name)

    def _build_model(self, input_shape, n_actions, learning_rate, model_name):
        """
        Builds the Tensorflow model.
        """
        self.model_name=model_name
        self.learning_rate = learning_rate
        self.model = build_dqn_model(model_name, input_shape, n_actions)
        self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate, 0.99)

    def predict(self, state):
        """
        Predicts action values.

        Args:
          state: State input of shape [batch_size, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, nb_actions] containing the estimated 
          action values.
        """
        return self.model.predict_on_batch(state)

    def update(self, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, 224, 224, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        loss_avg = tf.keras.metrics.Mean()
        for i in range(s.shape[0]):  # No sepuede hacertdoel batch a la vez???
            loss_value, grads = self._custom_grad(np.expand_dims(s[i], axis=0), y[i], a[i])
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            loss_avg.update_state(loss_value)
        return loss_avg.result()

    def _custom_loss(self, x, y, a):
        """
            custom loss function for dqn
            calculates loss only for the chosen action
        """
        y_ = self.model(x)
        y_ = y_[:, a]
        return huber(y, y_)

    def _custom_grad(self, inputs, targets, a):
        """
            custom grad function for dqn
            calculates gradient tape for custom loss
        """
        with tf.GradientTape() as tape:
            loss_value = self._custom_loss(inputs, targets, a=a)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def load_weights(self):
        self.model.load_weights("weights")

    def save_weights(self):
        self.model.save_weights("weights")

    def copy_weights(self, orig):
        orig.save_weights()
        self.load_weights()

    def save_model(self):
        self.model.save("dqn_model_"+self.model_name)

    def load_model(self,model_name):
        self.model = keras.models.load_model('dqn_model_'+model_name)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: number of actions


    Returns:
        A function that takes the (observation, epsilon,legal_actions) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation, epsilon, legal_actions):
        """
            policy function


                legal_actions: List of legal actions

        """

        A = np.zeros(nA, dtype=float)
        if len(legal_actions) == 0:
            return A
        for i in legal_actions:
            A[i] = epsilon / len(legal_actions)
        q_values = estimator.predict(tf.expand_dims(observation, axis=0))[0]
        best_actions = np.argsort(-q_values)
        for action in best_actions:
            if action in legal_actions:
                best_action = action
                break
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def deep_q_learning(env,
                    q_estimator,
                    target_estimator,
                    validation_env,
                    num_episodes,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    validate_every=1000,
                    rewards_mean_every=100,
                    discount_factor=1.0,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        num_episodes: Number of episodes to run for
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sample when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the 
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards. (PROVISIONAL)
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    stats = {}

    # The replay memory
    replay_memory = []
    total_t = 0

    # Keeps track of useful statistics (PROVISIONAL)
    stats["validation_episodes"] = []
    stats["validation_rewards"] = []
    stats["validation_hits"] = []
    stats["validation_class_changes"] = []
    stats["validation_positive_rewards"] = []

    stats["training_stats_episodes"] = []
    stats["training_rewards"] = []
    stats["training_losses"] = []
    stats["training_hits"] = []
    stats["training_total_steps"] = []
    stats["training_class_changes"] = []
    stats["training_positive_rewards"] = []

    cumulated_loss = 0
    cumulated_length = 0
    cumulated_reward = 0
    training_hits = 0
    training_class_changes = training_class_changes_bad = training_class_changes_good = training_class_changes_equal = 0
    training_positive_rewards = 0

    stats["num_episodes"] = num_episodes
    stats["learning_rate"] = q_estimator.learning_rate
    stats["env_info"] = "env max_steps:{} step_size:{} continue_until_dies:{}".format(env.max_steps, env.step_size,
                                                                                      env.continue_until_dies)
    stats["total_time"] = time.time()
    stats["validation_time"] = 0
    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        env.action_space.n)

    # Populate the replay memory with initial experience
    print("Training {} steps with LR= {}".format(num_episodes, q_estimator.learning_rate))
    print("Populating replay memory...")
    state = env.reset()
    # state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):
        legal_actions = env.get_legal_actions()
        if (len(legal_actions) == 0):
            print("ERROR: NO LEGAL ACTIONS POSSIBLE")
            break
        action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps - 1)], legal_actions)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        next_state, reward, done, _ = env.step(action)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
        else:
            state = next_state
    print("Done")

    # random_reward,random_hits=random_env_test(validation_env)
    # print("Random test on validation_env: validation reward mean: {} , hits: {}%".format(random_reward, random_hits))

    for i_episode in range(num_episodes):

        # print("Episode {}".format(i_episode))
        # Reset the environment
        state = env.reset()

        # One step in the environment
        ############# VALIDACION #######################
        if (i_episode + 1) % validate_every == 0:
            stats["validation_episodes"].append(i_episode)
            # print("Validating".format(i_episode))
            validation_time = time.time()
            validation_reward, hits, class_changes, class_changes_good, class_changes_bad, class_changes_equal, validation_positive_rewards = validation(
                q_estimator, validation_env)
            stats["validation_rewards"].append(float(validation_reward))
            stats["validation_hits"].append(hits)
            stats["validation_class_changes"].append(
                (class_changes, class_changes_good, class_changes_bad, class_changes_equal))
            stats["validation_positive_rewards"].append(validation_positive_rewards)
            print("\rEpisode {}/{}, validation_reward: {} hits: {} ".format(i_episode + 1, num_episodes,
                                                                            validation_reward, hits))
            validation_time = time.time() - validation_time
            stats["validation_time"] += validation_time
        ######################### ESTADISTICAS ###############
        if (i_episode + 1) % rewards_mean_every == 0:
            stats["training_stats_episodes"].append(i_episode)
            stats["training_rewards"].append(float(cumulated_reward / rewards_mean_every))
            stats["training_losses"].append(float(cumulated_loss / rewards_mean_every))  # numpy?)
            stats["training_total_steps"].append(float(cumulated_length / rewards_mean_every))
            stats["training_hits"].append(training_hits / rewards_mean_every)
            stats["training_class_changes"].append((training_class_changes / rewards_mean_every,
                                                    training_class_changes_bad / rewards_mean_every,
                                                    training_class_changes_good / rewards_mean_every,
                                                    training_class_changes_equal / rewards_mean_every))
            stats["training_positive_rewards"].append(training_positive_rewards / rewards_mean_every)
            cumulated_reward = cumulated_loss = cumulated_length = training_hits = 0
            training_class_changes = training_class_changes_bad = training_class_changes_good = training_class_changes_equal = 0
            training_positive_rewards = 0

        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            # Maybe update the target estimator
            if (total_t + 1) % update_target_estimator_every == 0:
                # print("Copying weights")
                target_estimator.copy_weights(q_estimator)
                gc.collect()
                # print("Copied")

            #################### INTERACCION CON EL ENV #########################
            # Take a step
            legal_actions = env.get_legal_actions()
            if len(legal_actions) == 0:
                print("ERROR: NO LEGAL ACTIONS POSSIBLE")
                break
            action_probs = policy(state, epsilon, legal_actions)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, info = env.step(action)
            if done:
                stats_reward = info["best_reward"]
                cumulated_reward += stats_reward
                episode_length = info["total_steps"]
                cumulated_length += episode_length
                class_change = info["class_change"]
                initial_hit = info["initial_hit"]
                hit = info["final_hit"]
                if hit:
                    training_hits += 1
                if class_change:
                    training_class_changes += 1
                    if hit:
                        training_class_changes_good += 1
                    else:
                        if initial_hit:
                            training_class_changes_bad += 1
                        else:
                            training_class_changes_equal += 1
                if (stats_reward > 0):
                    training_positive_rewards += 1

            ###################  GUARDADO EN MEMORIA ############################
            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            ################## APRENDIZAJE #############################
            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)

            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            q_values_next = target_estimator.predict(next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(
                q_values_next, axis=1)

            # Perform gradient descent update
            loss = q_estimator.update(states_batch, action_batch, targets_batch)
            gc.collect()

            cumulated_loss += loss
            total_t += 1

            ################ SIGUIENTE ESTADO / REINICIO ################
            if done:
                break
            state = next_state

    stats["total_time"] = time.time() - stats["total_time"]
    q_estimator.save_model()
    return stats
