from deep_q_learning_train import deep_q_learning_train

print("experiment 1. Deep Q learning step_size 32, max_Steps 5 continue until dies 1 intermediate rewards 1")
name=deep_q_learning_train(num_episodes=16000,learning_rate=0.00001,update_target_freq=120,validate_freq=4000,max_steps=5,
                                                                                    step_size=32,intermediate_rewards=1,continue_until_dies=1)
print("result: {}".format(name))
print("experiment 2. Deep Q learning step_size 32, max_Steps 1  intermediate rewards 1")
name=deep_q_learning_train(num_episodes=16000,learning_rate=0.00001,update_target_freq=120,validate_freq=4000,max_steps=12,
                                                                                    step_size=32,intermediate_rewards=1,continue_until_dies=0)
print("result: {}".format(name))