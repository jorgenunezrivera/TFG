from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2


def build_dqn_model(model_name,input_shape,n_actions,softmax_activation):
    if model_name== "atari":
        return build_atari_model(input_shape,n_actions,softmax_activation)
    elif model_name=="alexnet":
        return build_alexnet_model(input_shape,n_actions,softmax_activation)

    elif model_name=="pretrained_mobilenet":
        return build_mobilenet_model(input_shape,n_actions,softmax_activation)
    else:
        print("model name not recognized.using atari")
        return build_atari_model(input_shape, n_actions,softmax_activation)

def build_atari_model(input_shape,n_actions,softmax_activation):
    model_atari = keras.Sequential([
        layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
        # layers.MaxPooling2D(),
        layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        # layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu'),
        # layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(n_actions)
    ])
    if softmax_activation:
        model_atari.add(Activation('softmax'))
    return model_atari

def build_alexnet_model(input_shape,n_actions,softmax_activation):
    alexnet = Sequential()

    # Layer 1
    alexnet.add(layers.Conv2D(96, (11, 11), input_shape=input_shape,
                              padding='same', kernel_regularizer=l2(0)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(256, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(512, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_actions))
    alexnet.add(BatchNormalization())
    if softmax_activation:
       alexnet.add(Activation('softmax'))
    return alexnet

def build_mobilenet_model(input_shape,n_actions,softmax_activation):
    base_model = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(n_actions)
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = global_average_layer(x)
    x = keras.layers.Droput(0.2)

    x = prediction_layer(x)
    if softmax_activation:
       outputs=keras.layers.Activation('softmax')(x)
    else:
        outputs=x
    model = keras.model(inputs, outputs)
    return model
