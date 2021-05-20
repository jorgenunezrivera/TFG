import os

from enviroment_tests import check_dataset_posibilities
from deep_q_learning import deep_q_learning_train
from enviroment_tests import random_env_test
from reinforce import reinforce_train


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

print("experiment 2. Reinforce step_size 32, max_Steps 6 continue until dies 1 atari")
name=reinforce_train(num_episodes=12000,learning_rate=0.0000001,validate_freq=4000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="alexnet")
print("result: {}".format(name))

print("experiment 3. Rreinforce step_size 32, max_Steps 6 continue until dies 1 atari")
name=reinforce_train(num_episodes=12000,learning_rate=0.0000001,validate_freq=4000,max_steps=6,
                                                                                    step_size=32,continue_until_dies=1,model_name="pretrained_mobilenet")
print("result: {}".format(name))










