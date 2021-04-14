import os

from check_dataset_env import check_dataset_posibilities
from deep_q_learning_train import deep_q_learning_train
from random_env_test import random_env_test
from reinforce_train import reinforce_train
from window_env_generator import ImageWindowEnvGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

print("experiment 1. Deep Q learning step_size 32, max_Steps 6 continue until dies 1 alexnet")
name=deep_q_learning_train(num_episodes=10000,learning_rate=0.00001,update_target_freq=120,validate_freq=5000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="alexnet")
print("result: {}".format(name))


print("experiment 2. Deep Q learning step_size 32, max_Steps 6 continue until dies 1 pretrained_mobilenet")
name=deep_q_learning_train(num_episodes=10000,learning_rate=0.00001,update_target_freq=120,validate_freq=6000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="pretrained_mobilenet")
print("result: {}".format(name))


print("experiment 3. Reinforce step_size 32, max_Steps 6 continue until dies 1 alexnet")
name=reinforce_train(num_episodes=12000,learning_rate=0.00001,validate_freq=6000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="alexnet")
print("result: {}".format(name))

