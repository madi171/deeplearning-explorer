import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout

import tensorflow as tf
import numpy as np

model = Sequential([
    Conv2D(32, [5, 5], padding='same', input_shape=(28, 28, 1)),
    Activation('relu'),
    MaxPool2D((2, 2), strides=2),
    Conv2D(64, [5, 5], padding='same'),
    Activation('relu'),
    MaxPool2D((2, 2), strides=2),
    Flatten(),
    Dense(1024, input_shape=(784,)),
    Dropout(0.4),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_data = np.reshape(train_data, [train_data.shape[0], 28, 28, 1])
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
one_hot_labels = keras.utils.to_categorical(train_labels, num_classes=10)

model.fit(train_data, one_hot_labels, epochs=10, batch_size=32)
