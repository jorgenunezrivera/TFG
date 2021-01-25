import tensorflow as tf
from tensorflow import keras
import ssl

BATCH_SIZE = 3
IMG_SIZE = (160, 160)

IMG_SHAPE = IMG_SIZE + (3,)
model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=True, weights='imagenet')
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
()
model.save("mobilenet_v2_full.h5")
model.summary()
