import matplotlib.pyplot as plt
import sys
import json
import numpy as np


def training_plot(num_episodes,value_losses,action_losses,stats_mean_every,training_rewards,validation_rewards,validation_hits,action_stats):
    stats_x=list(range(stats_mean_every,num_episodes+1,stats_mean_every))
    plt.figure(figsize=(9, 9))
    plt.subplot(2, 2, 1)
    plt.plot(stats_x, action_losses, label='action losses')
    plt.plot(stats_x, value_losses, label='value losses')
    plt.legend(loc='upper right')
    plt.ylabel('Mean Squared Error')
    #plt.ylim([-1, 2])
    plt.title('Training Loss')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 2)
    plt.plot(stats_x, training_rewards,label='training_rewards')
    plt.plot(validation_rewards[0], validation_rewards[1], 'ro',label='validation_rewards')
    #plt.hlines(0,0,NUM_EPISODES)
    plt.legend(loc='upper right')
    plt.ylabel('Rewards')
    #plt.ylim([-1.1, 1.1])
    plt.title('Training Rewards')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 3)
    plt.plot(validation_hits[0], validation_hits[1])
   # plt.hlines(0.73,0,NUM_EPISODES)
    plt.legend(loc='upper right')
    plt.ylabel('Hits')
    #plt.ylim([0.65, 0.8])
    plt.title('Validation hits')
    plt.xlabel('epoch')

    plt.subplot(2, 2, 4)
    plt.plot(action_stats[0], action_stats[1],'b',label='action 0')
    plt.plot(action_stats[0], action_stats[2],'r',label='action 1')
    plt.plot(action_stats[0], action_stats[3],'g',label='action 2')
    if(action_stats[4]):
        plt.plot(action_stats[0], action_stats[4],'c',label='action 3')
    plt.legend(loc='upper right')
    plt.ylabel('Ocurrences')
    #plt.ylim([0, 5*num_episodes])
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
    training_plot(stats["num_episodes"],stats["value_losses"],stats["action_losses"],stats["stats_mean_every"],stats["total_returns"],stats["validation_rewards"],
                  stats["validation_hits"],stats["step_action"])
    print(stats)