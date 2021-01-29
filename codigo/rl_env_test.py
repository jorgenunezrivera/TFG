import gym
from image_window_env import ImageWindowEnv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import Memory
from rl.policy import EpsGreedyQPolicy
TEST_FILE="test.jpg"

image=tf.keras.preprocessing.image.load_img(TEST_FILE)
img_arr= keras.preprocessing.image.img_to_array(image)
#print(img_arr)
#print(img_arr.shape)
# The algorithms require a vectorized environment to run

env=ImageWindowEnv(img_arr)
#env = gym.make("BreakoutDeterministic-v4")
nb_actions = env.action_space.n

#

model = Sequential([
  layers.Reshape((160,160,3),input_shape=(1,160, 160,3)),
  #layers.experimental.preprocessing.Rescaling(1./255, input_shape=(160, 160, 3)),
  layers.Conv2D(16, (8,8),strides=(4,4), padding='same', activation='relu'),
  #layers.MaxPooling2D(),
  layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
  #layers.MaxPooling2D(),
  layers.GlobalAveragePooling2D(),  
  layers.Dense(16, activation='relu'),
  layers.Dense(nb_actions)
])
model.summary()

memory = Memory(window_length=1)
policy = EpsGreedyQPolicy(0.05)
rlagent = DQNAgent(model,enable_double_dqn=False,nb_actions=nb_actions,memory=memory,policy=policy)
rlagent.compile(Adam(lr=.05),metrics=['mae'])
rlagent.fit(env,1000,verbose=2)

rlagent.test(env,5)

obs= env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
