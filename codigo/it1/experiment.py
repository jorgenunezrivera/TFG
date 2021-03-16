from deep_q_learning_train import deep_q_learning_train

print("experiment 1. Deep Q learning step_size 32, max_Steps 1 continue until dies 1 intermediate rewards 2")
name=deep_q_learning_train(num_episodes=8000,learning_rate=0.00001,update_target_freq=120,validate_freq=4000,max_steps=1,
                                                                                    step_size=32,intermediate_rewards=2,continue_until_dies=1)
print("result: {}".format(name))
print("experiment 2. Deep Q learning step_size 32, max_Steps 2  continue_until_dies 1 intermediate rewards 1")
name=deep_q_learning_train(num_episodes=8000,learning_rate=0.00001,update_target_freq=120,validate_freq=4000,max_steps=2,
                                                                                    step_size=32,intermediate_rewards=2,continue_until_dies=1)
print("result: {}".format(name))

print("experiment 3. Deep Q learning step_size 32, max_Steps 3  continue_until_dies 1 intermediate rewards 1")
name=deep_q_learning_train(num_episodes=8000,learning_rate=0.00001,update_target_freq=120,validate_freq=4000,max_steps=3,
                                                                                    step_size=32,intermediate_rewards=2,continue_until_dies=1)
print("result: {}".format(name))

print("experiment 4. Deep Q learning step_size 32, max_Steps 4  continue_until_dies 1 intermediate rewards 1")
name=deep_q_learning_train(num_episodes=8000,learning_rate=0.00001,update_target_freq=120,validate_freq=4000,max_steps=4,
                                                                                    step_size=32,intermediate_rewards=2,continue_until_dies=1)
print("result: {}".format(name))