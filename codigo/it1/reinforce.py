import itertools

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from collections import namedtuple
from reinforce_validation import reinforce_validation

mse = tf.keras.losses.MeanSquaredError() #categoricalcrossentropy
mae = tf.keras.losses.MeanAbsoluteError()
huber=tf.keras.losses.Huber()

class PolicyEstimator():

    def __init__(self, input_shape, n_actions, learning_rate):
        self._build_model(input_shape, n_actions, learning_rate)

    def _build_model(self, input_shape, n_actions, learning_rate):
        """
        Builds the Tensorflow model.
        """
        self.learning_rate = learning_rate
        self.model = keras.Sequential([
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
            # layers.MaxPooling2D(),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            # layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu'),
            # layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(n_actions,activation='softmax')
        ])
        self.model.summary()
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
        picked_action_prob=action_probs[:,action]
        return -tf.math.log(picked_action_prob)*target

    def _custom_grad(self, state, target, action):
        with tf.GradientTape() as tape:
            loss_value = self._custom_loss(state, target, action)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def save_model(self):
        self.model.save("reinforce_policy_model")

class ValueEstimator():

    def __init__(self, input_shape, learning_rate):
        self._build_model(input_shape, learning_rate)

    def _build_model(self, input_shape, learning_rate):
        """
        Builds the Tensorflow model.
        """
        self.learning_rate = learning_rate
        self.model = keras.Sequential([
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
            # layers.MaxPooling2D(),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            # layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu'),
            # layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(1)
        ])
        self.model.summary()
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
        self.model.save("reinforce_value_model")


def reinforce(env, estimator_policy, estimator_value, num_episodes,validation_env, discount_factor=1.0,validate_every=200,stats_mean_every=100):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """


    stats={}
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    stats["value_losses"] = []
    stats["action_losses"] = []
    stats["total_returns"] = []
    stats["stats_mean_every"]=stats_mean_every
    stats["num_episodes"] = num_episodes
    stats["validation_rewards"] = []
    stats["validation_rewards"].append([])
    stats["validation_rewards"].append([])
    stats["validation_hits"] = []
    stats["validation_hits"].append([])
    stats["validation_hits"].append([])
    stats["action_stats"] = []
    cumulated_action_stats=np.zeros(env.action_space.n)
    stats["num_episodes"] = num_episodes
    stats["step_action"] = [[] for _ in range(5)]
    stats["policy_learning_rate"] = estimator_policy.learning_rate
    stats["value_learning_rate"] = estimator_value.learning_rate
    cumulated_value_loss = 0
    cumulated_action_loss = 0
    cumulated_return=0


    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        episode_value_loss=0
        episode_action_loss=0
        episode_total_return=0
        episode = []

######################## VALIDATION ###################
        if (i_episode + 1) % validate_every == 0:
            validation_reward, hits, wrong_certanty, action_stats = reinforce_validation(estimator_policy, validation_env)
            stats["validation_rewards"][0].append(i_episode)
            stats["validation_rewards"][1].append(float(validation_reward))
            stats["validation_hits"][0].append(i_episode)
            stats["validation_hits"][1].append(hits)
            cumulated_action_stats=np.add(cumulated_action_stats,action_stats)
            stats["step_action"][0].append(i_episode)
            for i in range(env.action_space.n):
                stats["step_action"][i + 1].append(action_stats[i])
            print("\rEpisode {}/{}, validation_reward: {} hits: {} mean_wrong_uncertanty: {}".format(i_episode + 1,
                                                                                                     num_episodes,
                                                                                                     validation_reward,
                                                                                                     hits,
                                                                                                     wrong_certanty))
####################STATS###################

        if (i_episode + 1) % stats_mean_every==0:
            cumulated_action_loss /= stats_mean_every
            cumulated_value_loss /= stats_mean_every
            cumulated_return /= stats_mean_every
            stats["value_losses"].append(cumulated_value_loss)
            stats["action_losses"].append(-cumulated_action_loss)
            stats["total_returns"].append(cumulated_return)
            cumulated_action_loss=cumulated_value_loss=cumulated_return=0

        # One step in the environment
        for t in itertools.count():

            # Take a step
            legal_actions=env.get_legal_actions()
            if len(legal_actions)==0:
                print("ERRO:NO POSSIBLE ACTIONS")
                break;
            action_probs = estimator_policy.predict((tf.expand_dims(state, axis=0)))[0]
            for i in range(len(action_probs)):
                if i not in legal_actions:
                    action_probs[i]=0
            action_probs=tf.nn.softmax(action_probs)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

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

        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this timestep
            total_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode[t:]))
            # Calculate baseline/advantage
            baseline_value = estimator_value.predict((tf.expand_dims(transition.state, axis=0)))[0]
            advantage = total_return - baseline_value
            # Update our value estimator
            episode_value_loss+=estimator_value.update(transition.state, total_return)
            # Update our policy estimator
            episode_action_loss+=estimator_policy.update(transition.state, transition.action,advantage)
            episode_total_return=total_return

        cumulated_action_loss+=float(episode_action_loss/len(episode))
        cumulated_value_loss+=float(episode_value_loss/len(episode))
        cumulated_return+=float(episode_total_return/len(episode))


    estimator_policy.save_model()
    estimator_value.save_model()
    stats["action_stats"] = cumulated_action_stats.tolist()
    return stats
