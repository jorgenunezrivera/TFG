import matplotlib.pyplot as plt
import sys
import json
import numpy as np

def training_plot(training_losses,training_rewards,validation_rewards,validation_hits,action_stats):
    training_losses_x = [x[0] for x in training_losses]
    training_losses_y = [x[1] for x in training_losses]
    training_rewards_x = [x[0] for x in training_rewards]
    training_rewards_y = [x[1] for x in training_rewards]
    validation_rewards_x = [x[0] for x in validation_rewards]
    validation_rewards_y = [x[1] for x in validation_rewards]
    validation_hits_x = [x[0] for x in validation_hits]
    validation_hits_y = [x[1] for x in validation_hits]

    plt.figure(figsize=(8, 8))
    plt.subplot(4, 1, 1)
    plt.plot(training_losses_x, training_losses_y)
    plt.legend(loc='upper right')
    plt.ylabel('Mean Absolute Error')
    #plt.ylim([0, 1])
    plt.title('Training Loss')
    plt.xlabel('epoch')

    plt.subplot(4, 1, 2)
    plt.plot(training_rewards_x, training_rewards_y,label='training_rewards')
    plt.plot(validation_rewards_x, validation_rewards_y, 'ro',label='validation_rewards')
    #plt.hlines(0,0,NUM_EPISODES)
    plt.legend(loc='upper right')
    plt.ylabel('Rewards')
    #plt.ylim([-1.1, 1.1])
    plt.title('Training Rewards')
    plt.xlabel('epoch')

    plt.subplot(4, 1, 3)
    plt.plot(validation_hits_x, validation_hits_y)
   # plt.hlines(0.73,0,NUM_EPISODES)
    plt.legend(loc='upper right')
    plt.ylabel('Hits')
    #plt.ylim([0.6, 0.9])
    plt.title('Validation hits')
    plt.xlabel('epoch')

    plt.subplot(4, 1, 4)
    plt.plot(action_stats[0], action_stats[1],'b',label='action 0')
    plt.plot(action_stats[0], action_stats[2],'r',label='action 1')
    plt.plot(action_stats[0], action_stats[3],'g',label='action 2')
    if len(action_stats[4]):
        plt.plot(action_stats[0], action_stats[4],'c',label='action 3')
    plt.legend(loc='upper right')
    plt.ylabel('Ocurrences')
    #plt.ylim([0, 6000])
    plt.title('Action stats')
    plt.xlabel('Action')

    plt.show()

    validation_reward_list = [x[1] for x in validation_rewards]
    validation_reward_mean = np.mean(validation_reward_list)
    validation_reward_variance = np.var(validation_reward_list)
    print("validation reward : mean: " + str(validation_reward_mean) + " variance: " + str(validation_reward_variance))





if len(sys.argv)!=2:
    print("Uso :  python plot_stats.py fichero_log.json")
else:
    with open(sys.argv[1], "r") as read_file:
        stats= json.load(read_file)
    training_plot(stats["training_losses"],stats["training_rewards"],stats["validation_rewards"],stats["validation_hits"],stats["step_action"])
    #print(stats)

