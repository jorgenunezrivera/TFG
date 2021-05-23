import os

from enviroment_tests import check_dataset_posibilities
from deep_q_learning import deep_q_learning_train
from enviroment_tests import random_env_test
from reinforce import reinforce_train


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

print("experiment 0. Fast reinforce atari")
name=reinforce_train(num_episodes=2000,learning_rate=0.0000001,validate_freq=1000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="atari")
print("result: {}".format(name))

print("experiment 1. DQN  atari")
name=deep_q_learning_train(num_episodes=2000,learning_rate=0.00001,validate_freq=1000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="atari")
print("result: {}".format(name))

print("experiment 2. DQN  alex")
name=deep_q_learning_train(num_episodes=2000,learning_rate=0.00001,validate_freq=1000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="alexnet")
print("result: {}".format(name))

print("experiment 3. DQN  mobilenet")
name=deep_q_learning_train(num_episodes=2000,learning_rate=0.00001,validate_freq=1000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="mobilenet")
print("result: {}".format(name))

print("experiment 4. Reinforce atari")
name=reinforce_train(num_episodes=12000,learning_rate=0.0000001,validate_freq=4000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="atari")
print("result: {}".format(name))

print("experiment 5. Reinforce alex")
name=reinforce_train(num_episodes=12000,learning_rate=0.0000001,validate_freq=4000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="alexnet")
print("result: {}".format(name))

print("experiment 6. Reinforce mobilenet")
name=reinforce_train(num_episodes=12000,learning_rate=0.0000001,validate_freq=4000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="mobilenet")
print("result: {}".format(name))











