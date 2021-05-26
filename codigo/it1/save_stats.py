import datetime
import json


def save_stats(stats, name):
    now = datetime.now()
    # print(stats)
    log_filename = now.strftime("logs/{}%d_%m_%Y_%H_%M_%S_log.json".format(name))
    with open(log_filename, 'w') as fp:
        json.dump(stats, fp)
        print("Saved to {}".format(log_filename))
    return log_filename