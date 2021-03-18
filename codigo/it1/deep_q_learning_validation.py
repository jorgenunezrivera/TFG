import numpy as np
import time
import itertools
import tensorflow as tf


def validation(q_estimator, env):
    #init_ts=time.time()
    rewards = []
    best_rewards=[]
    action_stats=np.zeros(env.action_space.n)
    hits=0
    incorrect_prediction_certainty=0
    for i in range(len(env)):
        obs = env.reset()
        for _ in itertools.count():
            q_values = q_estimator.predict(tf.expand_dims(obs, axis=0))[0]#NUMPY?
            legal_actions=env.get_legal_actions()
            best_actions = np.argsort(-q_values)
            for action in best_actions:
                if action in legal_actions:
                    best_action = action
                    break
            action_stats[best_action]+=1
            obs, reward, done, info = env.step(best_action)
            #if(i%20==0):
            #    print("q values: {}, reward: {} , hit:{}".format(q_values,reward,info["hit"]))
            if done:
                if info["best_hit"]:#cambiar best_hit por hit para rewards antiguos
                    hits += 1
                else:
                    incorrect_prediction_certainty+=info["max_prediction_value"]
                rewards.append(reward)
                best_rewards.append(info["best_reward"])
                break
    #print("time_elapsed={}".format(time.time()-init_ts))
    return np.mean(best_rewards),hits/len(env),incorrect_prediction_certainty/(len(env)-hits),action_stats #cambiar best_rewards por rewards para  rewards antiguos
