from deep_q_learning import deep_q_learning, Q_Estimator
from reinforce import PolicyEstimator, ValueEstimator, reinforce
from save_stats import save_stats
from window_env import ImageWindowEnv

TRAINING_IMAGES_DIR = "train_200"
VALIDATION_IMAGES_DIR = "validation1000"
TRAINING_LABELS_FILE = "training_labels.txt"
VALIDATION_LABELS_FILE = "validation_labels.txt"

env=ImageWindowEnv(TRAINING_IMAGES_DIR, TRAINING_LABELS_FILE)
env = ImageWindowEnv(TRAINING_IMAGES_DIR, TRAINING_LABELS_FILE, is_validation=0)

validation_env = ImageWindowEnv(VALIDATION_IMAGES_DIR, VALIDATION_LABELS_FILE, is_validation=1)



#deep Q Learning Atari
#q_estimator = Q_Estimator(model_name='atari')
#target_estimator = Q_Estimator(model_name='atari')
#stats = deep_q_learning(env, q_estimator, target_estimator, validation_env,  num_episodes=4000)
#save_stats(stats,'DQN_Atari')

#deep Q Learning Alexnet
#q_estimator = Q_Estimator(model_name='alexnet')
#target_estimator = Q_Estimator(model_name='alexnet')
#stats = deep_q_learning(env, q_estimator, target_estimator, validation_env, num_episodes=4000 )
#save_stats(stats,'DQN_Alexnet')

#deep Q Learning Mobilenet
q_estimator = Q_Estimator(model_name='mobilenet')
target_estimator = Q_Estimator(model_name='mobilenet')
stats = deep_q_learning(env, q_estimator, target_estimator, validation_env,num_episodes=2000,validate_every=1000,epsilon_decay_steps=8000)
save_stats(stats,'DQN_Mobilenet')

##REINFORCE

#Reinforce Atari

#policy_estimator = PolicyEstimator(model_name='atari')
#value_estimator = ValueEstimator(model_name='atari')

#stats=reinforce(env,policy_estimator,value_estimator,validation_env,num_episodes=12000,validate_every=4000,stats_mean_every=200)
#save_stats(stats,'REINFORCE_Atari')

#Reinforce Alexnet

#policy_estimator = PolicyEstimator(model_name='alexnet')
#value_estimator = ValueEstimator(model_name='alexnet')

#stats=reinforce(env,policy_estimator,value_estimator,validation_env,num_episodes=12000,validate_every=4000,stats_mean_every=200)
#save_stats(stats,'REINFORCE_Alexnet')

#Reinforce Mobilenet

#policy_estimator = PolicyEstimator(model_name='mobilenet')
#value_estimator = ValueEstimator(model_name='mobilenet')

#stats=reinforce(env,policy_estimator,value_estimator,validation_env,num_episodes=12000,validate_every=4000,stats_mean_every=200)
#save_stats(stats,'REINFORCE_Mobilenet')
