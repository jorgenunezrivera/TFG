import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
IMG_SIZE=(128,128)
IMG_SHAPE=(128,128,3)
model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=True,
                                               weights='imagenet')

#model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
#              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#              metrics=['accuracy'])

images=[]
test_set=[]
filenames=[]
files=os.listdir('validation')
for file in files:
    filename=os.path.join('validation', file)
    if os.path.isfile(filename) and filename.endswith('.JPEG'):
        images.append(tf.keras.preprocessing.image.load_img(filename))
        filenames.append(file)

for i in range(len(images)):
    array_img= keras.preprocessing.image.img_to_array(images[i])
    array_img=tf.image.resize(array_img,size=(IMG_SIZE))
    array_img = tf.keras.applications.mobilenet_v2.preprocess_input(array_img)
    test_set.append(array_img)
    

test_set=tf.stack(test_set)
predictions = model.predict_on_batch(test_set)
#predictions = tf.nn.sigmoid(predictionsTrained).numpy()
predictions_name=tf.keras.applications.mobilenet_v2.decode_predictions(predictions,1)
predictions_mean=0
#print("predictions name shape: {}".format(predictions_name.shape))    
for i in range(len(images)):
    predictions_mean+= predictions_name[i][0][2]
predictions_mean/=len(images)
print(predictions_mean)
