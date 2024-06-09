import tensorflow as tf
from tensorflow.keras import layers, models

def Net():
    model = models.Sequential([
        tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 1), strides=(2, 1), padding='valid', activation='relu'),
        #tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True),

        tf.keras.layers.Conv2D(filters=4, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        #tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True),

        tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        # tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 1)),

        tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 1)),

        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 1), strides=(2, 1), padding='valid', activation='relu'),
        # tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, trainable=True),
        # tf.keras.layers.MaxPool2D(pool_size=(2, 1)),

        layers.Flatten(),  
        layers.Dropout(0.5),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model



