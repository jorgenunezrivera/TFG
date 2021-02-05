import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
import numpy as np

import tensorflow as tf
from tensorflow import keras
HEIGHT=128
WIDTH=128
N_CHANNELS=3
MAX_STEPS=6
STEP_SIZE=8
N_ACTIONS=6


class ImageWindowEnvBatch(gym.Env):
    

    def __init__(self,img_arr_batch):
        super(ImageWindowEnvBatch, self).__init__()
        self.img_arr_batch=img_arr_batch
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space=spaces.Box(low=-1, high=1, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
        self.model=model = tf.keras.applications.MobileNetV2(input_shape=(HEIGHT, WIDTH, N_CHANNELS),
                                               include_top=True,
                                               weights='imagenet')
        self.sample_index=0
        self.num_samples=len(img_arr_batch)

    def reset(self):
        self.img_arr=self.img_arr_batch[self.sample_index]
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
        #0: up 1:right 2: down 3: left4: zoom in
        if action==0:
            self.y-=STEP_SIZE
        elif action==1:
            self.x+=STEP_SIZE
        elif action==2:
            self.y+=STEP_SIZE
        elif action==3:
            self.x-=STEP_SIZE
        elif action==4:
            self.z+=STEP_SIZE//2
        self.n_steps+=1
        state=self._get_image_window()
        predictions=self._get_predictions(state)
        self.predicted_class=self._get_predicted_class(predictions)        
        done=self.n_steps>=MAX_STEPS
        if done :
            reward=self.initial_reward-self._get_reward(predictions)
        else:
            reward=0#Reward parcial?
        return state,reward,done,{"predicted_class" : self.predicted_class}

    def render(self, mode='human', close=False):
        fig,ax=plt.subplots(1)
        ax.imshow(self.img_arr/255)
        rectangle=pltpatch.Rectangle((self.left,self.bottom),self.right-self.left,self.top-self.bottom,edgecolor='r',facecolor='none',linewidth=3)
        ax.add_patch(rectangle)
        #ax.imshow(self._get_image_window()/255)
        plt.show()
            

    def _get_image_window(self):
        self.left=(self.x+self.z)*self.image_size_factor[1]
        self.left=np.maximum(self.left,0)
        self.right=self.image_shape[1]+(self.x-self.z)*self.image_size_factor[1]
        self.right=np.minimum(self.right,self.image_shape[1])
        self.top=(self.y+self.z)*self.image_size_factor[0]
        self.top=np.maximum(self.top,0)
        self.bottom=self.image_shape[0]+(self.y-self.z)*self.image_size_factor[0]
        self.bottom=np.minimum(self.bottom,self.image_shape[0])
        image_window=self.img_arr[self.top:self.bottom,self.left:self.right]
        image_window_resized=tf.image.resize(image_window,size=(HEIGHT, WIDTH)).numpy()
        image_window_resized=tf.keras.applications.mobilenet_v2.preprocess_input(image_window_resized)
        return image_window_resized
   
    def _get_predictions(self,image_window):
        image_window_expanded=np.array([image_window])
        predictions=self.model.predict(image_window_expanded)
        return predictions

    def _get_predicted_class(self,predictions):
        prediction=tf.argmax(predictions,axis=1).numpy()[0]
        return prediction

    def _get_reward(self,predictions):
        reward = float(predictions[0,self.predicted_class])
        
        return reward
    
        
