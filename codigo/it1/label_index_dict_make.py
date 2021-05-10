import json
import tensorflow as tf
import numpy as np
import scipy

metadata=scipy.io.loadmat("meta.mat")
dictionary={}
for i in range(1000):
    dictionary[metadata['synsets'][i][0][1][0]]=int(metadata['synsets'][i][0][0][0][0])

indexes=[]
index_dict={}
for i in range(1000):
    fake_predictions=np.zeros(1000)
    fake_predictions[i]=1
    synset=tf.keras.applications.mobilenet_v2.decode_predictions(np.array([fake_predictions]), top=1)[0][0][0]
    label=dictionary[synset]
    index_dict[label]=i

print(indexes)
with open('label_to_index_dict.json', 'w') as fp:
    json.dump(index_dict, fp)