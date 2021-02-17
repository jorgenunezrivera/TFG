import tensorflow as tf
from window_env_batch import ImageWindowEnvBatch
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple
import itertools
import sys
import os
import random
import matplotlib.pyplot as plt
from deep_q_learning_validation import validation

mse = tf.keras.losses.MeanSquaredError() #categoricalcrossentropy
mae = tf.keras.losses.MeanAbsoluteError()
LEARNING_RATE = 0.00001

def custom_loss(model, x, y, training,a):
    y_ = model(x)
    #print("action: {} q_predictions: {} target_predictions: {}".format(a,y_,y))
    y_=y_[:,a]
    return mae(y,y_)
    
def custom_grad(model, inputs, targets,a):
    with tf.GradientTape() as tape:
        loss_value = custom_loss(model, inputs, targets, training=True,a=a)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def custom_train(model,optimizer,grad,s,y,a):
    loss_avg = tf.keras.metrics.Mean()
    for i in range(s.shape[0]): #No sepuede hacertdoel batch a la vez???
        loss_value, grads= custom_grad(model,np.expand_dims(s[i], axis=0),y[i],a[i])
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        loss_avg.update_state(loss_value)
    return loss_avg.result()

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self,input_shape,n_actions):
        self._build_model(input_shape,n_actions)

        
    def _build_model(self,input_shape,n_actions):
        """
        Builds the Tensorflow model.
        """
        self.model = keras.Sequential([
          layers.Conv2D(32, (8, 8), strides=(4, 4),  activation='relu', input_shape=input_shape),
          #layers.MaxPooling2D(),
          layers.Conv2D(64, (4, 4), strides=(2, 2),  activation='relu'),
          #layers.MaxPooling2D(),
          layers.Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'),
          #layers.MaxPooling2D(),
          layers.Flatten(),  
          layers.Dense(512, activation='relu'),
          layers.Dense(n_actions)#Tenia un softmax que no venia a cuento
        ])
        self.model.summary()
        self.optimizer=tf.keras.optimizers.RMSprop(LEARNING_RATE,0.99)

    def predict(self, state):
        """
        Predicts action values.

        Args:
          s: State input of shape [batch_size, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, nb_actions] containing the estimated 
          action values.
        """
        return self.model.predict(state)

    


    def update(self, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          s: State input of shape [batch_size, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        
        return custom_train(self.model,self.optimizer,custom_grad,s,y,a)

    def load_weights(self):
        self.model.load_weights("weights")

    def save_weights(self):
        self.model.save_weights("weights")

    def copy_weights(self, orig):
        orig.save_weights()
        self.load_weights()

    def save_model(self):
        self.model.save("dqn_model")

    def load_model(self):
        self.model = keras.models.load_model('dqn_model')

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
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
                    discount_factor=0.99,
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
        replay_memory_init_size: Number of random experiences to sampel when initializing 
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

    # The replay memory
    replay_memory = []
    total_t=0
    
    # Keeps track of useful statistics (PROVISIONAL)
    training_rewards=[]
    cumulated_reward=0
    training_losses=[]
    cumulated_loss=0
    validation_rewards=[]
    validation_hits=[]
    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        env.action_space.n)

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    #state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):
        action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        next_state, reward, done, _ = env.step(action)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
        else:
            state = next_state
    print("Done")

    for i_episode in range(num_episodes):

        
        # Reset the environment
        state = env.reset()
        #state = np.stack([state] * 4, axis=2)
        loss = None
        episode_loss=0
        # One step in the environment
        if (i_episode + 1) % validate_every == 0:
            validation_reward,hits,wrong_certanty = validation(q_estimator, validation_env)
            validation_rewards.append((i_episode, validation_reward))
            validation_hits.append((i_episode,hits))
            print("\rEpisode {}/{}, validation_reward: {} hits: {} mean_wrong_uncertanty: {}".format(i_episode + 1, num_episodes,validation_reward,hits,wrong_certanty))

        if (i_episode + 1) % rewards_mean_every==0:
            cumulated_reward/=rewards_mean_every
            training_rewards.append((i_episode,cumulated_reward))
            cumulated_loss/=rewards_mean_every
            training_losses.append((i_episode,cumulated_loss))
            cumulated_reward=cumulated_loss=0

        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Maybe update the target estimator
            if (total_t+1) % update_target_estimator_every == 0:
                target_estimator.copy_weights(q_estimator)

            # Take a step
            action_probs = policy(state, epsilon)            
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            #if (i_episode+1) % 20 == 0:#mostrar las acciones cada 20 samples
            #    print("action =" + str(action)+ " epsilon = " + str(epsilons[min(total_t, epsilon_decay_steps-1)]))
            next_state, reward, done, _ = env.step(action)
            #if (i_episode+1) % 20 == 0 and reward!=0:#mostrar las acciones cada 20 samples
            #    print("reward =" + str(reward))
             

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            q_values_next = target_estimator.predict(next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(states_batch, action_batch, targets_batch)
            episode_loss+=loss
            total_t += 1
            if done:
                cumulated_loss += episode_loss/(t+1)
                cumulated_reward += reward
                break

            state = next_state
            
  

    q_estimator.save_model()
    return training_losses, training_rewards ,validation_rewards, validation_hits


