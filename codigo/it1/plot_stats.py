import matplotlib.pyplot as plt
import sys
import json
import numpy as np

def training_plot(stats):
    training_episodes=stats["training_stats_episodes"]
    training_rewards=stats["training_rewards"]
    if 'training_losses' in stats:
        training_losses=stats["training_losses"]
    else:
        training_losses=(stats["training_value_losses"],stats["training_action_losses"])
    training_hits=stats["training_hits"]
    if np.max(training_hits)>1:
        training_hits[:] = [x/100 for x in training_hits]
    training_total_steps=stats["training_total_steps"]
    training_class_changes = [x[0] for x in stats["training_class_changes"]]
    training_class_better = [x[1] for x in stats["training_class_changes"]]
    training_class_worse = [x[2] for x in stats["training_class_changes"]]
    training_class_same = [x[3] for x in stats["training_class_changes"]]
    training_positive_rewards=stats["training_positive_rewards"]

    validation_episodes=stats["validation_episodes"]
    validation_rewards=stats["validation_rewards"]
    validation_hits=stats["validation_hits"]

    validation_class_changes = [x[0] for x in stats["validation_class_changes"]]
    validation_class_better = [x[1] for x in stats["validation_class_changes"]]
    validation_class_worse = [x[2] for x in stats["validation_class_changes"]]
    validation_class_same = [x[3] for x in stats["validation_class_changes"]]

    validation_positive_rewards=stats["validation_positive_rewards"]

    #Plot 1: training loss
    plt.figure(figsize=(16, 8))
    plt.subplot(3, 2, 1)
    if 'training_losses' in stats:
        plt.plot(training_episodes, training_losses,label='t.losses')
    else:
        plt.plot(training_episodes, training_losses[0],'r',label='v.losses')
        plt.plot(training_episodes, training_losses[1],'g',label='a.losses')
    plt.legend(loc='upper right')
    plt.ylabel('Huber Error')
    plt.ylim([5,35])
    plt.xlim([0, 2900])
    #plt.title('Training Loss')
    #plt.xlabel('epoch')

    #plot 2: trainig rewards and validation rewards
    plt.subplot(3, 2, 2)
    plt.plot(training_episodes, training_rewards,label='t.rewards')
    plt.plot(validation_episodes, validation_rewards, 'ro',label='v.rewards')
    plt.legend(loc='upper right')
    plt.ylabel('Rewards')
    plt.ylim([1.45, 2.1])
    plt.xlim([0, 2900])
    #plt.title('Rewards')
    #plt.xlabel('epoch')

    #plot 3: training hits and validation hits
    plt.subplot(3, 2, 3)
    plt.plot(training_episodes, training_hits,label='t.hits')
    plt.plot(validation_episodes, validation_hits,'ro',label='v.hits')
   # plt.hlines(0.73,0,NUM_EPISODES)
    plt.legend(loc='upper right')
    plt.ylabel('Hits')
    plt.ylim([0.55, 0.85])
    plt.xlim([0, 2900])
    #plt.title('Hits')
    #plt.xlabel('epoch')

    #plot 4: training and validation class changes
    plt.subplot(3, 2, 4)
    plt.plot(training_episodes, training_class_changes,'b',label='t. changes')
    plt.plot(training_episodes, training_class_better,'r',label='t. better')
    plt.plot(training_episodes, training_class_worse,'g',label='t. worse')
    #plt.plot(training_episodes, training_class_same, 'y', label='t. indiff')

    plt.plot(validation_episodes, validation_class_changes, 'bo', label='v. changes')
    plt.plot(validation_episodes, validation_class_better, 'ro',  label='v. better ')
    plt.plot(validation_episodes, validation_class_worse, 'go', label='v. worse')
    #plt.plot(validation_episodes, validation_class_same, 'y', linestyle='dashed', label='v.indif')


    plt.legend(loc='upper right')
    plt.ylabel('Percent')
    plt.ylim([0, 0.4])
    plt.xlim([0, 2900])
    #plt.title('Class changes')
    #plt.xlabel('epoch')

    #plot 5: positive rewards
    plt.subplot(3, 2, 5)
    plt.plot(training_episodes, training_positive_rewards, label='t.pos.rewards')
    plt.plot(validation_episodes,validation_positive_rewards,'ro',label='v.pos.rewards' )
    plt.legend(loc='upper right')
    plt.ylabel('Percent')
    plt.ylim([0.8, 0.92])
    plt.xlim([0, 2900])
    #plt.title('Class changes')
    #plt.xlabel('epoch')

    #plot 6: mean episode length
    plt.subplot(3, 2, 6)
    plt.plot(training_episodes, training_total_steps,label='ep.length')
    plt.legend(loc='upper right')
    plt.ylabel('Num steps')
    plt.ylim([9.7, 11.6])
    plt.xlim([0, 2900])
    #plt.title('Episode mean length')
    #plt.xlabel('epoch')
    plt.show()

    print(stats["env_info"])



if len(sys.argv)!=2:
    print("Uso :  python plot_stats.py fichero_log.json")
else:
    with open(sys.argv[1], "r") as read_file:
        stats= json.load(read_file)
    training_plot(stats)


