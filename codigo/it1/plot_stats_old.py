import matplotlib.pyplot as plt
import sys
import json
import numpy as np

def training_plot(training_losses,training_rewards,validation_rewards,validation_hits,action_stats,total_steps,env_info):
    training_rewards_x = [x[0] for x in training_rewards]
    training_rewards_y = [x[1] for x in training_rewards]
    validation_rewards_x = [x[0] for x in validation_rewards]
    validation_rewards_y = [x[1] for x in validation_rewards]
    validation_hits_x = [x[0] for x in validation_hits]
    validation_hits_y = [x[1] for x in validation_hits]
    if len(total_steps):
        total_steps_y=[x[1] for x in total_steps]

    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 2)
    plt.plot(training_rewards_x, training_rewards_y,label='training_rewards')
    plt.plot(validation_rewards_x, validation_rewards_y, 'ro',label='validation_rewards')
    #plt.hlines(0,0,NUM_EPISODES)
    plt.legend(loc='upper right')
    plt.ylabel('Rewards')
    plt.ylim([-1, 3])
    plt.title('Training Rewards')
    plt.xlabel('epoch')

    plt.subplot(1, 2, 1)
    plt.plot(validation_hits_x, validation_hits_y)
   # plt.hlines(0.73,0,NUM_EPISODES)
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')
    plt.ylim([0.58, 0.74])
    plt.title('Validation accuracy')
    plt.xlabel('epoch')




    plt.show()





if len(sys.argv)!=2:
    print("Uso :  python plot_stats.py fichero_log.json")
else:
    with open(sys.argv[1], "r") as read_file:
        stats= json.load(read_file)
    if not (stats.__contains__("total_steps")):
        stats["total_steps"]=[]
    if not (stats.__contains__("env_info")):
            stats["env_info"] = ""
    training_plot(stats["training_losses"],stats["training_rewards"],stats["validation_rewards"],stats["validation_hits"],stats["step_action"],stats["total_steps"],stats["env_info"])

    #print(stats)

