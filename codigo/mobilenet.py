import tensorflow as tf
from tensorflow import keras
import PIL
import PIL.Image
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

#Datos ( Perros y gatos)
print(os.getcwd())



train_dir = os.path.join('cats_and_dogs_filtered', 'train')
validation_dir = os.path.join('cats_and_dogs_filtered', 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
EPOCHS = 20

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
base_model = keras.models.load_model('mobilenet_v2.h5')

#base_model.summary()
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])


inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=EPOCHS,batch_size=32,
                    validation_data=validation_dataset)

model.save("filted_20_epochs_aumentation.h5")

#Test img
image = tf.keras.preprocessing.image.load_img('test.jpg')
image=tf.image.resize(image,size=(160, 160))
input_arr = keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
input_arr = input_arr/255
predictions = model.predict(input_arr)

print(predictions)
