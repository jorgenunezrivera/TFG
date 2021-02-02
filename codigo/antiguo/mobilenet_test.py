import tensorflow as tf
from tensorflow import keras
import PIL
import PIL.Image
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np


labels=["cat","dog"]

modelTrained = keras.models.load_model("fitted_10_epochs_aumentation_categorical")
modelTrained.summary()





#Test img
images=[]
test_set=[]
filenames=[]
files=os.listdir('test')
for file in files:
    filename=os.path.join('test', file)
    if os.path.isfile(filename) and filename.endswith('.jpg'):
        images.append(tf.keras.preprocessing.image.load_img(filename))
        filenames.append(file)

for i in range(len(images)):
    array_img= keras.preprocessing.image.img_to_array(images[i])
    array_img=tf.image.resize(array_img,size=(160, 160))
    #array_img = tf.keras.applications.mobilenet_v2.preprocess_input(array_img)
    test_set.append(array_img)
    

test_set=tf.stack(test_set)
plt.figure(figsize=(10, 10))
predictionsTrained = modelTrained.predict_on_batch(test_set)
predictions = tf.nn.sigmoid(predictionsTrained).numpy()
#logits = tf.where(predictions < 0.5, 0, 1).numpy()
#print(logits)
for i in range(len(images)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(" cat:"+str(predictions[i][0]) + " dog: " + str(predictions[i][1]))
plt.show()
print(predictions)
