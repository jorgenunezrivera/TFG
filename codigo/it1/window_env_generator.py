import gc

import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
import json
import from_disk_generator
from from_disk_generator import FromDiskGenerator

with open("label_to_index_dict.json", "r") as read_file:
    label_index_dict = json.load(read_file)

HEIGHT = 224
WIDTH = 224
N_CHANNELS = 3

MAX_STEPS = 18
STEP_SIZE = 32
N_ACTIONS = 4
INTERMEDIATE_REWARDS = 1
INTERMEDIATE_REWARDS_FACTOR = 10
CONTINUE_UNTIL_DIES = 0
if (CONTINUE_UNTIL_DIES):
    N_ACTIONS -= 1


class ImageWindowEnvGenerator(gym.Env):

    def __init__(self, directory, labels_file):
        super(ImageWindowEnvGenerator, self).__init__()
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
        self.last_better_result=-1
        self.max_possible_step=int(HEIGHT/STEP_SIZE)
        self.step_size=STEP_SIZE
        with open(labels_file) as fp:
            line = fp.readline()
            while line:
                self.labels.append(int(line))
                line = fp.readline()
        self.x = self.y = self.z = 0
        self.true_class = 0
        self.predicted_class = 0
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
        max_prediction_value = np.max(predictions)
        #print("Initial_rewrd: {}".format(self.initial_reward))
        self.history = [(0,0,0,self.initial_reward,self.predicted_class,max_prediction_value)]
        self.last_better_result = -1
        return image_window

    def restart_in_state(self,index):
        self.sample_index=index
        self.reset()

    def set_window(self,x,y,z):
        self.x=x
        self.y=y
        self.z=z


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

        if (self.x >= self.max_possible_step):
            self.x = self.max_possible_step-1
            self.n_steps =MAX_STEPS
        if (self.y >= self.max_possible_step):
            self.y = self.max_possible_step-1
            self.n_steps = MAX_STEPS
        if (self.z >= self.max_possible_step):
            self.z = self.max_possible_step-1
            self.n_steps = MAX_STEPS
        state = self._get_image_window()
        predictions = self._get_predictions(state)
        self.predicted_class = self._get_predicted_class(predictions)
        max_prediction_value = np.max(predictions)
        step_reward = self._get_reward(predictions)
        if CONTINUE_UNTIL_DIES:
            if step_reward <= self.history[self.last_better_result][3]:
                self.n_steps += 1
            else:
                self.last_better_result = len(self.history)
                self.n_steps = 0
        else:
            self.n_steps += 1
        done = (self.n_steps >= MAX_STEPS or action==3)

        if INTERMEDIATE_REWARDS:
            reward = (step_reward - self.history[-1][3]) * INTERMEDIATE_REWARDS_FACTOR
        else:
            if done:
                reward = (step_reward - self.initial_reward)*10
            else:
                reward=0
        topfive=keras.applications.mobilenet_v2.decode_predictions(predictions,5)
        self.history.append((self.x, self.y, self.z, step_reward, self.predicted_class, max_prediction_value))
        return state, reward, done, {"predicted_class": self.predicted_class,
                                     "max_prediction_value": max_prediction_value,
                                     "hit": (self.predicted_class == self.true_class),
                                     "top5": topfive}

    def render(self, mode='human', close=False):
        fig, ax = plt.subplots(1)
        ax.imshow(self.img_arr / 255)
        rectangle = pltpatch.Rectangle((self.left, self.bottom), self.right - self.left, self.top - self.bottom,
                                       edgecolor='r', facecolor='none', linewidth=3)
        ax.add_patch(rectangle)
        plt.show()

    def _get_image_window(self):
        self.left = (self.x) * self.image_size_factor[1] * STEP_SIZE
        self.left = np.maximum(self.left, 0)
        self.right = self.image_shape[1] + (self.x - self.z) * self.image_size_factor[1] * STEP_SIZE
        self.right = np.minimum(self.right, self.image_shape[1])
        self.top = (self.y) * self.image_size_factor[0] * STEP_SIZE
        self.top = np.maximum(self.top, 0)
        self.bottom = self.image_shape[0] + (self.y - self.z) * self.image_size_factor[0] * STEP_SIZE
        self.bottom = np.minimum(self.bottom, self.image_shape[0])
        image_window = Image.fromarray(np.uint8(self.img_arr[self.top:self.bottom, self.left:self.right]),mode='RGB')
        image_window_resized = image_window.resize((WIDTH,HEIGHT))#tf.image.resize(image_window, size=(HEIGHT, WIDTH))  # .numpy() (comprobar performance) #IMPLEMENTARSIN TENSORFLOW
        image_window_array=np.array(image_window_resized)#tf.keras.applications.mobilenet_v2.preprocess_input(image_window_resized)
        image_window_preprocessed=(image_window_array/128)-1
        return image_window_preprocessed

    def _get_predictions(self, image_window):
        predictions = self.model.predict_on_batch(tf.expand_dims(image_window, axis=0))#np.array
        gc.collect()
        return predictions

    def _get_predicted_class(self, predictions):
        predicted_class = np.argmax(predictions[0])
        return predicted_class

    def _get_reward(self, predictions):
        reward = float(predictions[0, self.true_class])
        return reward
