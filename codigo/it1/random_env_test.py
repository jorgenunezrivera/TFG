from window_env_generator import ImageWindowEnvGenerator
import numpy as np
import itertools
import json


def random_env_test(env):
    rewards = []
    hits=0
    for i in range(len(env)):
        env.reset()
        for t in itertools.count():
            legal_actions=env.get_legal_actions()
            rand_action = legal_actions[np.random.randint(len(legal_actions))]
            obs, reward, done, info = env.step(rand_action)
            if(done):
                if env.best_reward:
                    hit=info["best_hit"]
                    rewards.append(info["best_reward"])
                else:
                    hit=info["hit"]
                    rewards.append(reward)
                if(hit):
                    hits+=1
                break
    training_reward_mean=np.mean(rewards)
    return training_reward_mean,100*hits/len(env)

