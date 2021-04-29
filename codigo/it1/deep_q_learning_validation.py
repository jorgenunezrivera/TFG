import numpy as np
import time
import itertools
import tensorflow as tf


def validation(q_estimator, env):
    #init_ts=time.time()
    rewards = []
    hits=0
    class_changes=0
    class_changes_bad=0
    class_changes_good=0
    class_changes_equal=0
    positive_rewards=0
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
            obs, reward, done, info = env.step(best_action)
            #if(i%20==0):
            #    print("q values: {}, reward: {} , hit:{}".format(q_values,reward,info["hit"]))
            if done:
                class_change=info["class_change"]
                initial_hit=info["initial_hit"]
                hit=info["final_hit"]
                rewards.append(info["best_reward"])
                if hit:
                    hits += 1
                if class_change:
                    class_changes+=1
                    if hit:
                        class_changes_good+=1
                    else:
                        if initial_hit:
                            class_changes_bad+=1
                        else:
                            class_changes_equal+=1
                if info["best_reward"]>0:
                    positive_rewards+=1
                break
    #print("time_elapsed={}".format(time.time()-init_ts))
    return np.mean(rewards),hits/len(env),class_changes/len(env),class_changes_good/len(env),class_changes_bad/len(env),class_changes_equal/len(env),positive_rewards/len(env)
