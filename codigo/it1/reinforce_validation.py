import numpy as np
import time
import itertools


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
            chosen_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action_stats[chosen_action]+=1
            obs, reward, done, info = env.step(chosen_action)
            if(i%20==0):
                print("action_probs: {}, reward: {} , hit:{}".format(action_probs,reward,info["hit"]))
            if done:
                if info["hit"]:
                    hits += 1
                else:
                    incorrect_prediction_certainty+=info["max_prediction_value"]
                rewards.append(reward)
                break
    #print("time_elapsed={}".format(time.time()-init_ts))
    return np.mean(rewards),hits/len(env),incorrect_prediction_certainty/(len(env)-hits),action_stats
