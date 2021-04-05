from check_dataset_env import check_dataset_posibilities
from deep_q_learning_train import deep_q_learning_train
from random_env_test import random_env_test
from reinforce_train import reinforce_train
from window_env_generator import ImageWindowEnvGenerator

VALIDATION_IMAGES_DIR = "validation1000"
VALIDATION_LABELS_FILE = "validation_labels.txt"
env = ImageWindowEnvGenerator(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE,10,32,2,1,3,1)
random_reward,random_hits=random_env_test(env)
check_dataset_posibilities(env,5)

print("Random test. reward: {} hits: {}".format(random_reward,random_hits))


print("experiment 1. Reinforce step_size 32, max_Steps 6  continue_until_dies 1 intermediate rewards 2")
name=reinforce_train(num_episodes=12000,learning_rate=0.00001,validate_freq=4000,max_steps=6,
                                                                                    step_size=32,intermediate_rewards=2,continue_until_dies=1)
print("result: {}".format(name))
#

#
print("experiment 2. Reinforce step_size 32, max_Steps 10  continue_until_dies 0 intermediate rewards 2")
name=reinforce_train(num_episodes=12000,learning_rate=0.00001,validate_freq=4000,max_steps=10,
                                                                                    step_size=32,intermediate_rewards=2,continue_until_dies=0,n_actions=3)
print("result: {}".format(name))


print("experiment 3. Deep Q learning step_size 32, max_Steps 10  continue_until_dies 0 intermediate rewards 2 model=mobienet")
name=deep_q_learning_train(num_episodes=12000,learning_rate=0.00001,update_target_freq=120,validate_freq=4000,max_steps=10,
                                                                                    step_size=32,intermediate_rewards=2,
                                                                                    continue_until_dies=0,n_actions=3,model_name="pretrained_mobilenet")
print("result: {}".format(name))

print("experiment 4. Deep Q learning step_size 32, max_Steps 6 continue until dies 1 intermediate rewards 2 model=mobilenet ")
name=deep_q_learning_train(num_episodes=12000,learning_rate=0.00001,update_target_freq=120,validate_freq=4000,max_steps=6,
                                                                                    step_size=32,intermediate_rewards=2,continue_until_dies=1,model_name="pretrained_mobilenet")
print("result: {}".format(name))
