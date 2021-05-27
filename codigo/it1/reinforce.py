import itertools
import math
import json
import os

from window_env import ImageWindowEnv

from datetime import datetime
import time

from tensorflow import keras
import tensorflow as tf
import numpy as np
from collections import namedtuple

from build_models import build_dqn_model

mse = tf.keras.losses.MeanSquaredError() #categoricalcrossentropy
mae = tf.keras.losses.MeanAbsoluteError()
huber=tf.keras.losses.Huber()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# IMG_SHAPE=(224,224,3)
TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"

NUM_EPISODES = 12000
LEARNING_RATE = 0.000001
VALIDATE_EVERY=3000
MODEL_NAME="atari"

#env default params
#Default parameters
N_ACTIONS=3
MAX_STEPS = 6
STEP_SIZE = 32
CONTINUE_UNTIL_DIES = 1


def reinforce_train(num_episodes=NUM_EPISODES,learning_rate=LEARNING_RATE,
                          validate_freq=VALIDATE_EVERY,max_steps=MAX_STEPS,step_size=STEP_SIZE,
                 continue_until_dies=CONTINUE_UNTIL_DIES,model_name=MODEL_NAME):

    env = ImageWindowEnv(TRAINING_IMAGES_DIR, TRAINING_LABELS_FILE, max_steps, step_size,
                                  continue_until_dies,is_validation=0)

    validation_env = ImageWindowEnv(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE, max_steps, step_size,
                                             continue_until_dies,is_validation=1)

    N_ACTIONS = env.action_space.n
    IMG_SHAPE = env.observation_space.shape

    initial_ts = time.time()

    policy_estimator = PolicyEstimator(IMG_SHAPE, N_ACTIONS, learning_rate,model_name)
    value_estimator = ValueEstimator(IMG_SHAPE, learning_rate,model_name)

    stats=reinforce(env,policy_estimator,value_estimator,validation_env,num_episodes=num_episodes,validate_every=validate_freq,stats_mean_every=200)

    elapsed_time = time.time() - initial_ts
    print("Elapsed time: " + str(elapsed_time))
    print("Num episodes: " + str(NUM_EPISODES))
    print("secs/episode:" + str(elapsed_time / NUM_EPISODES))

    now = datetime.now()
    #print(stats)
    log_filename = now.strftime("logs/%d_%m_%Y_%H_%M_%S_reinforce_log.json")
    with open(log_filename, 'w') as fp:
        json.dump(stats, fp)
    return log_filename

def reinforce_validation(action_estimator, env):
    #init_ts=time.time()
    rewards = []
    hits=0
    class_changes = 0
    class_changes_bad = 0
    class_changes_good = 0
    class_changes_equal = 0
    positive_rewards = 0
    for i in range(len(env)):
        obs = env.reset()
        for _ in itertools.count():
            action_probs = action_estimator.predict(np.array([obs]))[0]
            action_probs = tf.nn.softmax(action_probs).numpy()
            legal_actions = env.get_legal_actions()
            for i in range(len(action_probs)):
                if i not in legal_actions:
                    action_probs[i] = 0
            if np.sum(action_probs)==0 or math.isnan(sum(action_probs)):
                print("action probs error: sum action_probs =0")
                break;
            action_probs =action_probs/np.sum(action_probs)
            chosen_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            obs, reward, done, info = env.step(chosen_action)
            if done:
                class_change = info["class_change"]
                initial_hit=info["initial_hit"]
                hit=info["final_hit"]
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
    #print("time_elapsed={}".format(time.time()-init_ts))
    return np.mean(rewards),hits/len(env),class_changes/len(env),class_changes_good/len(env),class_changes_bad/len(env),class_changes_equal/len(env),positive_rewards/len(env)


class PolicyEstimator():

    def __init__(self, input_shape=(224,224,3), n_actions=3, learning_rate=.0000001,model_name='atari'):
        self._build_model(input_shape, n_actions, learning_rate,model_name)

    def _build_model(self, input_shape, n_actions, learning_rate,model_name):
        """
        Builds the Tensorflow model.
        """
        self.model_name=model_name
        self.learning_rate = learning_rate
        self.model = build_dqn_model(model_name, input_shape, n_actions)
        self.optimizer = keras.optimizers.RMSprop(self.learning_rate, 0.99)

    def predict(self, state):
        """
        Predicts action values.

        Args:
          s: State input of shape [batch_size, 224, 224, 3]

        Returns:
          Tensor of shape [batch_size, nb_actions] containing the estimated
          action values.
        """
        return self.model.predict(state)

    def update(self, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [224, 224, 3]
          a: Chosen actions of shape [1]
          y: Targets of shape [1]

        Returns:
          The calculated loss on the batch.
        """
        loss_value, grads = self._custom_grad(s, y, a)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    def _custom_loss(self, state, target, action):
        action_probs = self.model(tf.expand_dims(state,axis=0))
        action_probs = tf.nn.softmax(action_probs)
        picked_action_prob=action_probs[:,action]
        return -tf.math.log(picked_action_prob)*target

    def _custom_grad(self, state, target, action):
        with tf.GradientTape() as tape:
            loss_value = self._custom_loss(state, target, action)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def save_model(self):
        self.model.save("reinforce_policy_model_"+self.model_name)

    def load_model(self,model_name):
        self.model = keras.models.load_model('reinforce_policy_model_'+model_name)

class ValueEstimator():

    def __init__(self, input_shape, learning_rate,model_name):
        self._build_model(input_shape, learning_rate,model_name)

    def _build_model(self, input_shape, learning_rate,model_name):
        """
        Builds the Tensorflow model.
        """
        self.model_name=model_name
        self.learning_rate = learning_rate
        self.model = build_dqn_model(model_name, input_shape, 1)
        self.optimizer = keras.optimizers.RMSprop(self.learning_rate, 0.99)

    def predict(self, state):
        """    
        Predicts action values.

        Args:
          s: State input of shape [224, 224, 3]

        Returns:
          Tensor of shape [batch_size, nb_actions] containing the estimated
          action values.
        """
        return self.model(state)

    def update(self, s, y):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, 224, 224, 3]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        loss_value, grads = self._custom_grad(s, y)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value

    def _custom_loss(self, state, target):
        value_estimate = self.model(np.expand_dims(state, axis=0))
        target=tf.expand_dims(target, axis=0)
        return mse(value_estimate,target)

    def _custom_grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self._custom_loss(inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def save_model(self):
        self.model.save("reinforce_value_model_"+self.model_name)


def reinforce(env, policy_estimator, value_estimator,validation_env, num_episodes=12000, discount_factor=1.0
            ,validate_every=4000,stats_mean_every=200):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
        validate_every: Number of episodes between validations
        stats_mean_every: Number of episodes between stats mean calculation

    Returns:
        A stats dictionary
    """


    stats={}
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    stats["training_stats_episodes"] = []
    stats["training_value_losses"] = []
    stats["training_action_losses"] = []
    stats["training_rewards"] = []
    stats["training_hits"]=[]
    stats["training_total_steps"] = []
    stats["training_class_changes"] = []
    stats["training_positive_rewards"] = []

    stats["validation_episodes"]=[]
    stats["validation_rewards"] = []
    stats["validation_hits"] = []
    stats["validation_class_changes"] = []
    stats["validation_positive_rewards"] = []


    stats["num_episodes"] = num_episodes
    stats["policy_learning_rate"] = policy_estimator.learning_rate
    stats["value_learning_rate"] = value_estimator.learning_rate

    cumulated_value_loss = 0
    cumulated_action_loss = 0
    cumulated_return=0
    cumulated_length=0
    training_hits = 0
    training_class_changes = training_class_changes_bad = training_class_changes_good = training_class_changes_equal = 0
    training_positive_rewards = 0

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        episode_value_loss=0
        episode_action_loss=0
        episode_total_return=0
        episode = []

######################## VALIDATION ###################
        if (i_episode + 1) % validate_every == 0:
            validation_reward, hits,class_changes,class_changes_good,class_changes_bad,class_changes_equal,validation_positive_rewards  = reinforce_validation(policy_estimator, validation_env)
            stats["validation_episodes"].append(i_episode)
            stats["validation_rewards"].append(float(validation_reward))
            stats["validation_hits"].append(hits)
            stats["validation_class_changes"].append(
                (class_changes, class_changes_good, class_changes_bad, class_changes_equal))
            stats["validation_positive_rewards"].append(validation_positive_rewards)
            print("\rEpisode {}/{}, validation_reward: {} hits: {} ".format(i_episode + 1,
                                                                                                     num_episodes,
                                                                                                     validation_reward,
                                                                                                     hits
                                                                                                     ))
####################STATS###################

        if (i_episode + 1) % stats_mean_every==0:
            stats["training_stats_episodes"].append(i_episode)
            stats["training_value_losses"].append(cumulated_value_loss/stats_mean_every)
            stats["training_action_losses"].append(-cumulated_action_loss/stats_mean_every)
            stats["training_rewards"].append(cumulated_return/stats_mean_every)
            stats["training_total_steps"].append(float(cumulated_length / stats_mean_every))
            stats["training_hits"].append(training_hits / stats_mean_every)
            stats["training_class_changes"].append((training_class_changes / stats_mean_every,
                                                    training_class_changes_good / stats_mean_every,
                                                    training_class_changes_bad / stats_mean_every,
                                                    training_class_changes_equal / stats_mean_every))
            stats["training_positive_rewards"].append(training_positive_rewards / stats_mean_every)
            cumulated_action_loss=cumulated_value_loss=cumulated_return=cumulated_length=0
            training_hits=0
            training_class_changes = training_class_changes_bad = training_class_changes_good = training_class_changes_equal = 0
            training_positive_rewards = 0

        # One step in the environment
        for t in itertools.count():

            # Take a step
            legal_actions=env.get_legal_actions()
            if len(legal_actions)==0:
                print("ERRO:NO POSSIBLE ACTIONS")
                break;
            elif len(legal_actions)==1:
                action_probs=np.zeros(env.action_space.n)
                action_probs[legal_actions[0]]=1
            else:
                action_probs = policy_estimator.predict((tf.expand_dims(state, axis=0)))[0]
                action_probs = tf.nn.softmax(action_probs).numpy()
                for i in range(len(action_probs)):
                    if i not in legal_actions:
                        action_probs[i]=0
            #Que pasa aqui
            if(np.sum(action_probs))==0 or math.isnan(np.sum(action_probs)):
                print("ERRO:NO CHOSEN LEGAL ACTIONS : {}".format(action_probs))
                break;
            action_probs=action_probs/np.sum(action_probs)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, info = env.step(action)



            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, reward=reward, next_state=next_state, done=done))

            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, reward), end="")
            # sys.stdout.flush()

            if done:

                break

            state = next_state
        if len(episode)==0:
            break
        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode[t:]))
            # Calculate baseline/advantage
            baseline_value = value_estimator.predict((tf.expand_dims(transition.state, axis=0)))[0]
            advantage = total_return - baseline_value
            # Update our value estimator
            episode_value_loss+=value_estimator.update(transition.state, total_return)
            # Update our policy estimator
            step_action_loss=policy_estimator.update(transition.state, transition.action,advantage)
            episode_action_loss+=step_action_loss

        ######## ACTUALIZAR ESTADISTICAS DEL EPISODIO ##################
        cumulated_action_loss+=float(episode_action_loss/len(episode))
        cumulated_value_loss+=float(episode_value_loss/len(episode))
        stats_reward=float(info["best_reward"])
        cumulated_return += stats_reward
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


    policy_estimator.save_model()
    value_estimator.save_model()
    return stats
