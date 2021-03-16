import gc

import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
import numpy as np
from PIL import Image

import tensorflow as tf
import json
import from_disk_generator
from from_disk_generator import FromDiskGenerator

with open("label_to_index_dict.json", "r") as read_file:
    label_index_dict = json.load(read_file)

# Fixed Hiperparams (constants)

HEIGHT = 224
WIDTH = 224
N_CHANNELS = 3
N_ACTIONS = 4

REWARDS_FACTOR = 10

# Default parameters
MAX_STEPS = 6
STEP_SIZE = 32

INTERMEDIATE_REWARDS = 0
CONTINUE_UNTIL_DIES = 0


# Legal actions


class ImageWindowEnvGenerator(gym.Env):

    def __init__(self, directory, labels_file, max_steps=MAX_STEPS, step_size=STEP_SIZE,
                 intermediate_rewards=INTERMEDIATE_REWARDS,
                 continue_until_dies=CONTINUE_UNTIL_DIES, n_actions=N_ACTIONS):
        super(ImageWindowEnvGenerator, self).__init__()
        self.max_steps = max_steps
        self.step_size = step_size
        self.intermediate_rewards = intermediate_rewards
        self.continue_until_dies = continue_until_dies
        self.n_actions=n_actions
        if self.continue_until_dies:
            self.n_actions -= 1
        image_filenames = from_disk_generator.get_filenames(directory)
        self.image_generator = FromDiskGenerator(
            image_filenames, batch_size=1,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
        self.model = tf.keras.applications.MobileNetV2(input_shape=(HEIGHT, WIDTH, N_CHANNELS),
                                                       include_top=True,
                                                       weights='imagenet')
        self.sample_index = 0
        self.num_samples = self.image_generator.__len__()
        self.labels = []
        self.last_better_result = -1
        self.max_possible_step = int(HEIGHT / STEP_SIZE)
        self.step_size = STEP_SIZE
        with open(labels_file) as fp:
            line = fp.readline()
            while line:
                self.labels.append(int(line))
                line = fp.readline()
        self.x = self.y = self.z = 0
        self.true_class = 0
        self.predicted_class = 0
        self.best_result=0
        self.best_predicted_class=-1
        self.history = []

    def __len__(self):
        return self.num_samples

    def reset(self):
        self.img_arr = self.image_generator.__getitem__([self.sample_index])[0]
        label = self.labels[self.sample_index]
        self.true_class = label_index_dict[str(label)]
        self.sample_index += 1  # Batches siempre en el mismo orden???
        if (self.sample_index >= self.num_samples):  # >=?
            self.sample_index = 0
        self.image_shape = self.img_arr.shape
        self.image_size_factor = (self.image_shape[0] // HEIGHT, self.image_shape[1] // WIDTH)
        self.x = self.y = self.z = 0
        self.left = self.right = self.top = self.bottom = 0
        self.n_steps = 0

        image_window = self._get_image_window()
        predictions = self._get_predictions(image_window)
        self.predicted_class = self._get_predicted_class(predictions)
        self.initial_reward = self._get_reward(predictions)
        self.best_result=self.initial_reward
        self.best_predicted_class=-1

        max_prediction_value = np.max(predictions)
        # print("Initial_rewrd: {}".format(self.initial_reward))
        self.history = [(0, 0, 0, self.initial_reward, self.predicted_class, max_prediction_value)]
        self.last_better_result = -1
        return image_window

    def restart_in_state(self, index):
        self.sample_index = index
        self.reset()

    def set_window(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_legal_actions(self):
        actions = []
        if self.x < self.z:
            actions.append(0)
        if self.y < self.z:
            actions.append(1)
        if self.z < self.max_possible_step - 1:
            actions.append(2)
        if  self.n_actions == 4:
            actions.append(3)
        return actions

    def step(self, action):
        # 0: right 1:down 2: zoom in 3: end
        if action == 0:
            self.x += 1
        elif action == 1:
            self.y += 1
        elif action == 2:
            self.z += 1
        elif action == 3:
            pass

        state = self._get_image_window()
        predictions = self._get_predictions(state)
        self.predicted_class = self._get_predicted_class(predictions)
        max_prediction_value = np.max(predictions)
        step_reward = self._get_reward(predictions)

        if self.continue_until_dies:
            if step_reward <= self.best_result:
                self.n_steps += 1
            else:
                self.n_steps = 0
        else:
            self.n_steps += 1

        done = (self.n_steps >= self.max_steps or action == 3 or len(self.get_legal_actions())==0)

        if self.intermediate_rewards==1:
            reward = (step_reward - self.history[-1][3]) * REWARDS_FACTOR
        elif self.intermediate_rewards==2:
            reward=(step_reward-self.best_result) * REWARDS_FACTOR
        else:
            if done:
                reward = (step_reward - self.initial_reward) * REWARDS_FACTOR
            else:
                reward = 0

        if step_reward > self.best_result:
            self.best_result = step_reward
            self.best_predicted_class=self.predicted_class

        self.history.append((self.x, self.y, self.z, step_reward, self.predicted_class, max_prediction_value))
        return state, reward, done, {"predicted_class": self.predicted_class,
                                     "max_prediction_value": max_prediction_value,
                                     "hit": (self.predicted_class == self.true_class),
                                     "best_hit": (self.best_predicted_class == self.true_class)}

    def render(self, mode='human', close=False):
        fig, ax = plt.subplots(1)
        ax.imshow(self.img_arr / 255)
        rectangle = pltpatch.Rectangle((self.left, self.bottom), self.right - self.left, self.top - self.bottom,
                                       edgecolor='r', facecolor='none', linewidth=3)
        ax.add_patch(rectangle)
        plt.show()

    def _get_image_window(self):
        self.left = (self.x) * self.image_size_factor[1] * self.step_size
        self.left = np.maximum(self.left, 0)
        self.right = self.image_shape[1] + (self.x - self.z) * self.image_size_factor[1] * self.step_size
        self.right = np.minimum(self.right, self.image_shape[1])
        self.top = (self.y) * self.image_size_factor[0] * self.step_size
        self.top = np.maximum(self.top, 0)
        self.bottom = self.image_shape[0] + (self.y - self.z) * self.image_size_factor[0] * self.step_size
        self.bottom = np.minimum(self.bottom, self.image_shape[0])
        image_window = Image.fromarray(np.uint8(self.img_arr[self.top:self.bottom, self.left:self.right]), mode='RGB')
        image_window_array = image_window.resize((WIDTH,
                                                    HEIGHT))  # tf.image.resize(image_window, size=(HEIGHT, WIDTH))  # .numpy() (comprobar performance) #IMPLEMENTARSIN TENSORFLOW
        image_window_array = np.array(
            image_window_array)  # tf.keras.applications.mobilenet_v2.preprocess_input(image_window_resized)
        image_window_array = np.add(image_window_array, -128)
        image_window_array = np.divide(image_window_array,128,dtype=np.float16)
        return image_window_array

    def _get_predictions(self, image_window):
        predictions = self.model.predict_on_batch(np.expand_dims(image_window, axis=0))  # np.array
        gc.collect()
        return predictions

    def _get_predicted_class(self, predictions):
        predicted_class = np.argmax(predictions[0])
        return predicted_class

    def _get_reward(self, predictions):
        reward = float(predictions[0, self.true_class])
        return reward

    #for test purposes
    def random_window(self):
        z=np.random.randint(1,self.max_possible_step)
        x=np.random.randint(z)
        y=np.random.randint(z)
        self.set_window(x,y,z)
        state= self._get_image_window()
        predictions = self._get_predictions(state)
        return predictions