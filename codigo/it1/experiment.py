from deep_q_learning_train import deep_q_learning_train
from reinforce_train import reinforce_train

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
