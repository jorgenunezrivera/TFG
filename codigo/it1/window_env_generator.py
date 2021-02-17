import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
import numpy as np

import tensorflow as tf
from tensorflow import keras
import json
import from_disk_generator
from from_disk_generator import FromDiskGenerator

with open("label_to_index_dict.json", "r") as read_file:
    label_index_dict = json.load(read_file)

HEIGHT=224
WIDTH=224
N_CHANNELS=3
MAX_STEPS=5
STEP_SIZE=16
N_ACTIONS=4
REWARD_MAXIMIZING=0

class ImageWindowEnvGenerator(gym.Env):
    

    def __init__(self,directory,labels):
        super(ImageWindowEnvBatch, self).__init__()
        image_filenames=from_disk_generator.get_filenames(directory)
        self.image_generator = FromDiskGenerator(
            image_filenames, batch_size=1,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space=spaces.Box(low=-1, high=1, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
        self.model= tf.keras.applications.MobileNetV2(input_shape=(HEIGHT, WIDTH, N_CHANNELS),
                                               include_top=True,
                                               weights='imagenet')
        self.sample_index=0
        self.num_samples=self.image_generator.__len__()
        self.labels=labels
        self.history=[] #(x,y,z,return)


    def reset(self):
        self.img_arr=self.image_generator.__get_item__(self.sample_index)
        label=self.labels[self.sample_index]
        self.true_class=label_index_dict[str(label)]
        self.sample_index+=1#Batches siempre en el mismo orden???
        if(self.sample_index>=self.num_samples):#>=?
            self.sample_index=0            
        self.image_shape=self.img_arr.shape
        self.image_size_factor=(self.image_shape[0]//HEIGHT,self.image_shape[1]//WIDTH)
        self.x=self.y=self.z=0
        self.left=self.right=self.top=self.bottom=0
        self.n_steps=0
        image_window=self._get_image_window()
        predictions=self._get_predictions(image_window)
        self.predicted_class=self._get_predicted_class(predictions)        
        self.initial_reward=self._get_reward(predictions)
        return image_window

    def step (self,action):
        #0: right 1:down 2: zoom in 3: end
        if action==0:
            self.x+=STEP_SIZE
        elif action==1:
            self.y+=STEP_SIZE
        elif action==2:
            self.z+=STEP_SIZE
        elif action==3:
            pass
        self.n_steps+=1
        state=self._get_image_window()
        predictions=self._get_predictions(state)
        predicted_class=self._get_predicted_class(predictions)
        max_prediction_value=np.max(predictions)
        step_reward = self._get_reward(predictions)
        done=(self.n_steps>=MAX_STEPS or action==3)
        self.history.append((self.x,self.y,self.z,step_reward))
        if done :
            reward = step_reward - self.initial_reward
            #self.cumulated_rewards.append(reward)
            #variance=np.var(self.cumulated_rewards)
            if(REWARD_MAXIMIZING):
                if(reward>0):
                    reward=1
                if(reward<0):
                    reward=-1
            #elif(REWARD_NORMALIZATION):
            #    if(variance!=0):
            #        reward = reward/variance

        else:
            reward=0#Reward parcial?
        return state,reward,done,{"predicted_class" : predicted_class, "max_predition_value":max_prediction_value}

    def render(self, mode='human', close=False):
        fig,ax=plt.subplots(1)
        ax.imshow(self.img_arr/255)
        rectangle=pltpatch.Rectangle((self.left,self.bottom),self.right-self.left,self.top-self.bottom,edgecolor='r',facecolor='none',linewidth=3)
        ax.add_patch(rectangle)
        plt.show()

    def _get_image_window(self):
        self.left=(self.x)*self.image_size_factor[1]
        self.left=np.maximum(self.left,0)
        self.right=self.image_shape[1]+(self.x-self.z)*self.image_size_factor[1]
        self.right=np.minimum(self.right,self.image_shape[1])
        self.top=(self.y)*self.image_size_factor[0]
        self.top=np.maximum(self.top,0)
        self.bottom=self.image_shape[0]+(self.y-self.z)*self.image_size_factor[0]
        self.bottom=np.minimum(self.bottom,self.image_shape[0])
        image_window=self.img_arr[self.top:self.bottom,self.left:self.right]
        image_window_resized=tf.image.resize(image_window,size=(HEIGHT, WIDTH))#.numpy() (comprobar performance)
        image_window_resized=tf.keras.applications.mobilenet_v2.preprocess_input(image_window_resized)
        return image_window_resized
   
    def _get_predictions(self,image_window):
        image_window_expanded=np.array([image_window])
        predictions=self.model.predict(image_window_expanded)
        return predictions

    def _get_predicted_class(self,predictions):
        predicted_class=np.argmax(predictions[0])
        return predicted_class

    def _get_reward(self,predictions):
        reward = float(predictions[0,self.true_class])
        return reward


        
