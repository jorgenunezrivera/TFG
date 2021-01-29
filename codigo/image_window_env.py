import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
import numpy as np

import tensorflow as tf
from tensorflow import keras
HEIGHT=160
WIDTH=160
N_CHANNELS=3
MAX_STEPS=10
STEP_SIZE=10

class ImageWindowEnv(gym.Env):
    

    def __init__(self,img_arr):
        super(ImageWindowEnv, self).__init__()
        self.x=self.y=self.z=0
        self.left=self.right=self.top=self.bottom=0
        self.img_arr=img_arr
        self.action_space = spaces.Discrete(5)
        self.n_steps=0
        self.observation_space=spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.model=keras.models.load_model("fitted_10_epochs_aumentation_categorical")


    def reset(self):
        self.image_shape=self.img_arr.shape
        self.image_size_factor=(self.image_shape[0]//160,self.image_shape[1]//160)
        self.x=self.y=self.z=0
        self.n_steps=0
        return self._get_image_window()

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
            self.z+=STEP_SIZE
        self.n_steps+=1
        state=self._get_image_window()
        reward=self._get_reward(state)
        done=self.n_steps>=MAX_STEPS
        return state,reward,done,{}

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
        image_window_resized=tf.image.resize(image_window,size=(160, 160)).numpy()
        #image_window_resized=self._test_image_resize(image_window,size=(160, 160))
        return image_window_resized

    def _get_reward(self,image_window):
        #image_window_expanded=tf.expand_dims(image_window, 0)
        image_window_expanded=np.array([image_window])
        predictions=self.model.predict(image_window_expanded)
        predictions=tf.nn.sigmoid(predictions).numpy()
        reward = tf.math.reduce_max(predictions).numpy() #NO TIENE SENTIDO PERO POR AHORA
        return float(reward)
    
    def _test_image_resize(self,image,size=(160,160)):#NO TIENE SENTIDO PERO POR AHORA
        height, width,_ = image.shape
        if(size[0]<width and size[1]<height):
            resized=image[:size[0],:size[1]]
        else:
            resized=np.zeros((size[0],size[1]))
            for i in range (width):
                for j in range (height):
                    resized[i][j]=image[i][j]
        return resized    


        
