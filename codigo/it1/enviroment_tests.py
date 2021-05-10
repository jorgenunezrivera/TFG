import numpy as np
import itertools


def random_env_test(env):
    """
        Tests an enviroment acting randomly
        Args:
          env : ImageWindowEnv enviroment to test
        Returns:
          float,  mean of the rewards from every sample
          int?, percent of samples with correct prediction (hits)
    """
    rewards = []
    hits=0
    for i in range(len(env)):
        env.reset()
        for t in itertools.count():
            legal_actions=env.get_legal_actions()
            rand_action = legal_actions[np.random.randint(len(legal_actions))]
            obs, reward, done, info = env.step(rand_action)
            if(done):
                hit=info["final_hit"]
                rewards.append(info["best_reward"])
                if(hit):
                    hits+=1
                break
    reward_mean=np.mean(rewards)
    return reward_mean,100*hits/len(env)

def random_env_test_batch(env,n):
    """
        tests an environment acting randomly for a number of times, gets the mean and max results
        Args:
            env : ImageWindowEnv environment to be tested
            n: int, number of times the enviroment will be checked

    """
    cumulated_random_reward = 0
    cumulated_random_hits = 0
    max_hits=0
    for i in range(n):
        random_reward, random_hits = random_env_test(env)
        if(random_hits>max_hits):
            max_hits=random_hits
        cumulated_random_reward += random_reward
        cumulated_random_hits += random_hits
    random_hits = cumulated_random_hits/n
    random_reward = cumulated_random_reward / n
    print("{} random tests. mean reward: {} mean hits: {} max hits:{}".format(n,random_reward, random_hits,max_hits))
    return random_reward,random_hits,max_hits

def sliding_window(index, env,max_zoom):
    env.restart_in_state(index)
    for z in range(max_zoom):
        for y in range(z):
            for x in range(z):
                # print("sample {}  x,y,z  ={},{},{}".format(index, x, y, z))
                if(x+y+z)>env.max_steps:
                    break
                env.set_window(x, y, z)
                _, _, _, info = env.step(3)
                if (info["hit"]):
                    #print("sample {} hit with x, y,z  ={},{},{}, certainty:{}".format(index, x, y, z,
                    #                                                                  info["max_prediction_value"]))
                    return True
    #print("sample {} wrong".format(index))
    return False


def check_dataset_posibilities(env,max_zoom):
    wrongs = []
    hits = 0
    for i in range(len(env)):
        _ = env.reset()
        _, _, _, info = env.step(3)
        hit = info["hit"]
        if not hit:
            wrongs.append(i)
        else:
            hits += 1

    #print("Hits: {}/{} ({}%)".format(hits, len(env), hits * 100 / len(env)))

    fixable_wrongs = []
    for w in wrongs:
        if sliding_window(w, env,max_zoom):
            fixable_wrongs.append(w)

    print(
        " {} hits, {} wrongs, {} fixable wrongs with step size: {} and MAX_STEPS: {}, max precission:{}".format(
            hits, len(wrongs), len(fixable_wrongs), env.step_size, env.max_steps, (hits + len(fixable_wrongs)) * 100 / len(env)))
    return (hits + len(fixable_wrongs)) * 100 / len(env)
