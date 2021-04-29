import math

import numpy as np
import time
import itertools
import tensorflow as tf


def reinforce_validation(action_estimator, env):
    #init_ts=time.time()
    rewards = []
    hits=0
    class_changes = 0
    class_changes_bad = 0
    class_changes_good = 0
    class_changes_equal = 0
    positive_rewards = 0
    for i in range(len(env)):
        obs = env.reset()
        for _ in itertools.count():
            action_probs = action_estimator.predict(np.array([obs]))[0]
            action_probs = tf.nn.softmax(action_probs).numpy()
            legal_actions = env.get_legal_actions()
            for i in range(len(action_probs)):
                if i not in legal_actions:
                    action_probs[i] = 0
            if np.sum(action_probs)==0 or math.isnan(sum(action_probs)):
                print("action probs error: sum action_probs =0")
                break;
            action_probs =action_probs/np.sum(action_probs)
            chosen_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            obs, reward, done, info = env.step(chosen_action)
            if done:
                class_change = info["class_change"]
                initial_hit=info["initial_hit"]
                hit=info["final_hit"]
                rewards.append(info["best_reward"])
                if hit:
                    hits += 1
                if class_change:
                    class_changes += 1
                    if hit:
                        class_changes_good += 1
                    else:
                        if initial_hit:
                            class_changes_bad += 1
                        else:
                            class_changes_equal += 1
                if info["best_reward"] > 0:
                    positive_rewards += 1
                break
    #print("time_elapsed={}".format(time.time()-init_ts))
    return np.mean(rewards),hits/len(env),class_changes/len(env),class_changes_good/len(env),class_changes_bad/len(env),class_changes_equal/len(env),positive_rewards/len(env)
