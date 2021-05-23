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
N_ACTIONS = 3

REWARDS_FACTOR = 10

# Default parameters
MAX_STEPS = 6
STEP_SIZE = 32

CONTINUE_UNTIL_DIES = 0
BEST_REWARD=1
IS_VALIDATION=0


class ImageWindowEnv(gym.Env):

    def __init__(self, directory, labels_file, max_steps=MAX_STEPS, step_size=STEP_SIZE,
                 continue_until_dies=CONTINUE_UNTIL_DIES, is_validation=IS_VALIDATION):
        super(ImageWindowEnv, self).__init__()
        self.max_steps = max_steps
        self.step_size = step_size
        self.continue_until_dies = continue_until_dies
        self.n_actions=N_ACTIONS
        self.is_validation=is_validation
        image_filenames = from_disk_generator.get_filenames(directory)
        self.image_generator = FromDiskGenerator(
            image_filenames, batch_size=1,
        )
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
        self.model = tf.keras.applications.MobileNetV2(input_shape=(HEIGHT, WIDTH, N_CHANNELS),
                                                       include_top=True,
                                                       weights='imagenet')
        self.sample_index = 0
        self.num_samples = self.image_generator.__len__()
        self.labels = []
        self.max_possible_step = int(HEIGHT / STEP_SIZE)
        with open(labels_file) as fp:
            line = fp.readline()
            while line:
                self.labels.append(int(line))
                line = fp.readline()
        self.x = self.y = self.z = 0
        self.true_class = 0
        self.predicted_class = 0
        self.best_result=0
        self.best_stop_result=0
        self.best_predicted_class=-1
        self.total_reward=0
        self.left = self.right = self.top = self.bottom = 0
        self.n_steps = 0
        self.img_arr = 0
        self.image_shape = 0
        self.image_size_factor = 0
        self.initial_reward=0
        self.initial_stop_reward=0
        self.initial_prediction=0

    def __len__(self):
        return self.num_samples

    def reset(self):
        """
            Carga una nueva imagen, situa la ventana en el origen y reinicia las variables de cálculo de recompensa

        """
        #Carga una nueva imagen
        self.img_arr = self.image_generator.__getitem__([self.sample_index])[0]
        label = self.labels[self.sample_index]
        self.true_class = label_index_dict[str(label)]
        self.sample_index += 1  # Batches siempre en el mismo orden???
        if self.sample_index >= self.num_samples:  # >=?
            self.sample_index = 0
        self.image_shape = self.img_arr.shape
        self.image_size_factor = (self.image_shape[0] // HEIGHT, self.image_shape[1] // WIDTH)
        #Situa la ventana en el origen
        self.x = self.y = self.z = 0
        self.left = self.right = self.top = self.bottom = 0
        self.n_steps = 0

        #Reinicia las variables de calculo de recompensas
        image_window = self._get_image_window()
        predictions = self._get_predictions(image_window)
        self.predicted_class = self._get_predicted_class(predictions)
        self.initial_prediction=self.predicted_class
        self.initial_reward = self._get_reward(predictions,True)
        self.initial_stop_reward=self._get_reward(predictions,self.is_validation)
        self.best_result=self.initial_reward
        self.best_stop_result=self.initial_stop_reward
        self.best_predicted_class = self.predicted_class
        #self.total_reward=self.initial_reward
        return image_window



    def step(self, action):
        """avanza un paso la posición de la ventana,
            luego recalcula el estado y las recompensas

            Args:
                action : int(0-3)

            Returns:
                a tuple (state, reward,done,info}
                state is the next state
                reward is the training reward,
                done indicates if the state is final
                info is a dictionary wich contains validation information
        """
        #Avanza un paso
        # 0: right 1:down 2: zoom in 3: end
        if action == 0:
            self.x += 1
        elif action == 1:
            self.y += 1
        elif action == 2:
            self.z += 1
        elif action == 3:
            pass

        #Actualiza el estado y los rewards
        state = self._get_image_window()
        predictions = self._get_predictions(state)
        self.predicted_class = self._get_predicted_class(predictions)
        step_reward = self._get_reward(predictions,True)
        stop_reward = self._get_reward(predictions, self.is_validation)

        best_reward=0
        class_change=0
        initial_hit=0
        final_hit=0

        #Comprueba  si el estado es final
        if self.continue_until_dies:
            if stop_reward <= self.best_stop_result:
                self.n_steps += 1
            else:
                self.n_steps = 0
        else:
            self.n_steps += 1

        done = (self.n_steps >= self.max_steps or action == 3 or len(self.get_legal_actions())==0)


        #Calcula los rewards
        #REWARD USADO PARA ENTRENAR
        #reward=(step_reward-self.best_result) * REWARDS_FACTOR #CALCULA EL REWARD USANDO LA CLASE REAL
        reward = (stop_reward - self.best_stop_result) * REWARDS_FACTOR #CALCULA EL REWARD USANDO EL MEJOR RESULTADO

        if step_reward >= self.best_result:
            self.best_result = step_reward
            #self.best_predicted_class=self.predicted_class  #CALCULA EL MEJOR RESULTADO USANDO LA CLASE REAL

        if stop_reward >= self.best_stop_result:#Mayor o igual?? cero??
            self.best_stop_result = stop_reward
            self.best_predicted_class = self.predicted_class #CALCULA EL MEJOR RESULTADO USANDO EL VALOR MÁXIMO DE CONFIANZA DE LA RED

        if self.is_validation and step_reward != stop_reward:
            print("step_reward: {}".format(step_reward))
            print("stop_reward: {}".format(stop_reward))

        if done:
            #REWARD USADO PARA FINES INFORMATIVOS
            best_reward = (self.best_result - self.initial_reward) * REWARDS_FACTOR
            class_change= self.best_predicted_class!=self.initial_prediction
            initial_hit = self.initial_prediction==self.true_class
            final_hit=self.best_predicted_class==self.true_class
            #print("initial prediction. {} final prediction. {}".format(self.initial_prediction,self.best_predicted_class))
        hit=self.predicted_class==self.true_class
        self.total_reward += reward
        return state, reward, done, {"hit":hit,
                                     "initial_hit": initial_hit,
                                     "final_hit": final_hit,
                                     "best_reward": best_reward,
                                     "class_change":class_change,
                                     "total_steps":self.x+self.y+self.z,
                                     "position":(self.x,self.y,self.z)}

    def render(self, mode='human', close=False):
        """Muestra la imagen con la ventana dibujada"""
        fig, ax = plt.subplots(1)
        ax.imshow(self.img_arr / 255)
        rectangle = pltpatch.Rectangle((self.left, self.bottom), self.right - self.left, self.top - self.bottom,
                                       edgecolor='r', facecolor='none', linewidth=3)
        ax.add_patch(rectangle)
        plt.show()

    def get_legal_actions(self):
        """devuelve una lista con las acciones legales"""
        actions = []
        if self.x < self.z:
            actions.append(0)
        if self.y < self.z:
            actions.append(1)
        if self.z < self.max_possible_step - 1:
            actions.append(2)
        if self.n_actions == 4:
            actions.append(3)
        return actions

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

    def _get_reward(self, predictions,validation):
        if validation:
            reward=np.max(predictions[0])
        else:
            reward = float(predictions[0, self.true_class])
        return reward

    def restart_in_state(self, index):
        self.sample_index = index
        self.reset()

    def set_window(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self._get_image_window()


