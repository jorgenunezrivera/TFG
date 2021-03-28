import numpy as np
import time
import itertools
import tensorflow as tf


def reinforce_validation(action_estimator, env):
    #init_ts=time.time()
    rewards = []
    action_stats=np.zeros(env.action_space.n)
    hits=0
    incorrect_prediction_certainty=0
    for i in range(len(env)):
        obs = env.reset()
        for _ in itertools.count():
            action_probs = action_estimator.predict(np.array([obs]))[0]
            action_probs = tf.nn.softmax(action_probs).numpy()
            legal_actions = env.get_legal_actions()
            for i in range(len(action_probs)):
                if i not in legal_actions:
                    action_probs[i] = 0
            if np.sum(action_probs)==0:
                print("action probs error: sum action_probs =0")
                break;
            action_probs =action_probs/np.sum(action_probs)
            chosen_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action_stats[chosen_action]+=1
            obs, reward, done, info = env.step(chosen_action)
            if(i%20==0):
                print("action_probs: {}, reward: {} , hit:{}".format(action_probs,reward,info["hit"]))
            if done:
                if(env.best_reward):
                    hit=info["best_hit"]
                    rewards.append(info["best_reward"])
                else:
                    hit=info["hit"]
                    rewards.append(reward)
                if hit:
                    hits += 1
                else:
                    incorrect_prediction_certainty+=info["max_prediction_value"]
                break
    #print("time_elapsed={}".format(time.time()-init_ts))
    return np.mean(rewards),hits/len(env),incorrect_prediction_certainty/(len(env)-hits),action_stats#cambiar np.mean(best_rewards por np.mean (rewards para reward final
