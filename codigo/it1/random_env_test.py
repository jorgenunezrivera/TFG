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
                    rewards.append(info["total_reward"])
                if(hit):
                    hits+=1
                break
    training_reward_mean=np.mean(rewards)
    return training_reward_mean,100*hits/len(env)

def random_env_test_batch(env,n):
    cumulated_random_reward = 0
    cumulated_random_hits = 0
    max_hits=0
    for i in range(n):
        random_reward, random_hits = random_env_test(env)
        if(random_hits>max_hits):
            max_hits=random_hits
        cumulated_random_reward += random_reward
        cumulated_random_hits += random_hits
    cumulated_random_hits /= 10
    cumulated_random_reward /= 10
    print("{} random tests. mean reward: {} mean hits: {} max hits:{}".format(n,cumulated_random_reward, cumulated_random_hits,max_hits))

