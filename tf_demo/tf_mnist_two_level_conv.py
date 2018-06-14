# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Input, Flatten
from keras.optimizers import Adam

mnist = input_data.read_data_sets("datasets/MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(10, activation='relu'))

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=50, epochs=5, verbose=1)
