from window_env_generator import ImageWindowEnvGenerator
import numpy as np
import itertools
import json


def random_env_test(env):
    rewards = []
    hits=0
    N_ACTIONS = env.action_space.n
    for i in range(len(env)):
        env.reset()
        env.render
        for t in itertools.count():
            best_action = np.random.randint(N_ACTIONS)
            obs, reward, done, info = env.step(best_action)
            if(done):
                rewards.append(reward)
                if(info["hit"]):
                    hits+=1
                break
    training_reward_mean=np.mean(rewards)
    return training_reward_mean,100*hits/len(env)

