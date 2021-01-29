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

def test_image(filename,x=0,y=0,zoom=0):
    if zoom>79:
        zoom=79
    image=tf.keras.preprocessing.image.load_img(filename)
    array_img= keras.preprocessing.image.img_to_array(image)
    #array_img=tf.image.resize(array_img,size=(160, 160))
    image_shape=array_img.shape
    image_size_factor=(image_shape[0]//160,image_shape[1]//160)
    left=(x+zoom)*image_size_factor[0]
    right=image_shape[0]+(x-zoom)*image_size_factor[0]
    right=np.minimum(right,image_shape[0])
    top=(y+zoom)*image_size_factor[1]
    bottom=image_shape[1]+(y-zoom)*image_size_factor[1]
    bottom=np.minimum(bottom,image_shape[1])
    image_window=array_img[left:right,top:bottom]
    image_window_resized=tf.image.resize(image_window,size=(160, 160))
    image_window_resized = tf.expand_dims(image_window_resized, 0) # Create a batch         
    plt.figure(figsize=(10, 10))
    predictions = modelTrained.predict(image_window_resized)
    predictions = tf.nn.sigmoid(predictions).numpy()
    plt.imshow(image_window_resized[0]/255)
    plt.title(" cat:"+str(predictions[0][0]) + " dog: " + str(predictions[0][1]))
    #plt.show()    
    return predictions[0]

print(test_image("test.jpg"))
print(test_image("test.jpg",10,10,20))
print(test_image("test.jpg",10,10,30))
    


