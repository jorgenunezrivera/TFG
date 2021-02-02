import tensorflow as tf
from tensorflow import keras
import ssl

BATCH_SIZE = 3
IMG_SIZE = (160, 160)

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False, weights='imagenet')
base_model.save("mobilenet_v2.h5")
base_model.summary()
