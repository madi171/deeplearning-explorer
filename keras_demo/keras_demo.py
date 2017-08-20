from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model = Sequential()

X_train = np.random.random((1000, 1000))
print X_train
Y_train = np.random.random((1000, 10))
print Y_train

model.add(Dense(units=6400, input_dim=1000))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=500, batch_size=32)
