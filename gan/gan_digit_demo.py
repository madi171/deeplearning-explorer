import matplotlib

matplotlib.use('Agg')

import os

os.putenv("CUDA_VISIBLE_DEVICES", "0")
# os.system("bash")
print "Env set OK!"

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer, Dropout, LeakyReLU, Conv2D, Activation, \
    BatchNormalization, UpSampling2D, Conv2DTranspose
from keras.optimizers import RMSprop
from keras.regularizers import L1L2

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

train = pd.read_csv('datasets/train.csv')
train = train.drop(['label'], axis=1)
X = train.as_matrix()
X = X / 255.
X = X.reshape(X.shape[0], 28, 28)

latend_dim = 100
dropout = 0.4

# ---------- standart gan --------------
model_G = Sequential([
    Dense(units=500, input_dim=latend_dim, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Dropout(dropout),
    Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Dropout(dropout),
    Dense(units=784, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Reshape((28, 28))
])

model_G.summary()

model_D = Sequential([
    InputLayer(input_shape=(28, 28)),
    Flatten(),
    Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Dropout(dropout),
    Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
    Dropout(dropout),
    Dense(units=1, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5))
])

model_D.summary()

# -------- model with leakyReLU ----------
# model_G = Sequential([
#     Dense(units=500, input_dim=latend_dim, activation=LeakyReLU(0.2), kernel_regularizer=L1L2(1e-5, 1e-5)),
#     Dropout(dropout),
#     Dense(units=500, activation=LeakyReLU(0.2), kernel_regularizer=L1L2(1e-5, 1e-5)),
#     Dropout(dropout),
#     Dense(units=784, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)),
#     Reshape((28, 28))
# ])
#
# model_G.summary()
#
# model_D = Sequential([
#     InputLayer(input_shape=(28, 28)),
#     Flatten(),
#     Dense(units=500, activation=LeakyReLU(0.2), kernel_regularizer=L1L2(1e-5, 1e-5)),
#     Dropout(dropout),
#     Dense(units=500, activation=LeakyReLU(0.2), kernel_regularizer=L1L2(1e-5, 1e-5)),
#     Dropout(dropout),
#     Dense(units=1, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5))
# ])
#
# model_D.summary()


# -------- DCGAN ----------
# X = X.reshape(X.shape[0], 28, 28, 1)
# depth_G=4*64
# depth_D=64
# dim = 7
# model_G = Sequential([
#
#     Dense(dim * dim * depth_G, input_dim=100),
#     BatchNormalization(momentum=0.9),
#     Activation('relu'),
#     Reshape((dim, dim, depth_G)),
#     Dropout(dropout),
#
#     UpSampling2D(),
#     Conv2DTranspose(int(depth_G/2), 5, padding='same'),
#     BatchNormalization(momentum=0.9),
#     Activation('relu'),
#
#     Conv2DTranspose(int(depth_G/4), 5, padding='same'),
#     BatchNormalization(momentum=0.9),
#     Activation('relu'),
#
#     UpSampling2D(),
#     Conv2DTranspose(int(depth_G/8), 5, padding='same'),
#     BatchNormalization(momentum=0.9),
#     Activation('relu'),
#
#     Conv2DTranspose(1, 5, padding='same'),
#     Activation('sigmoid'),
# ])
#
# model_G.summary()
#
# model_D = Sequential([
#     InputLayer(input_shape=(28, 28, 1)),
#
#     Conv2D(depth_D * 1, 5, strides=2, padding='same'),
#     LeakyReLU(alpha=0.2),
#     Dropout(dropout),
#
#     Conv2D(depth_D * 2, 5, strides=2, padding='same'),
#     LeakyReLU(alpha=0.2),
#     Dropout(dropout),
#
#     Conv2D(depth_D * 4, 5, strides=2, padding='same'),
#     LeakyReLU(alpha=0.2),
#     Dropout(dropout),
#
#     Conv2D(depth_D * 8, 5, strides=2, padding='same'),
#     LeakyReLU(alpha=0.2),
#     Dropout(dropout),
#
#     Flatten(),
#     Dense(1, activation='sigmoid')
#
# ])
#
# model_D.summary()



# ------------ self made gan ---------------
d_model = Sequential()
d_model.add(model_D)
d_model.compile(loss="binary_crossentropy", optimizer=RMSprop(lr=0.0002), metrics=['accuracy'])
print "dmodel"
d_model.summary()

ad_model = Sequential()
ad_model.add(model_G)
model_D.trainable = False
ad_model.add(model_D)
ad_model.compile(loss="binary_crossentropy", optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])

print "admodel"
ad_model.summary()

batch_size = 128
num_epochs = 10000

for i in xrange(num_epochs):
    image_true = X[np.random.randint(0, X.shape[0], size=batch_size)]
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latend_dim])
    image_fake = model_G.predict(noise)
    x = np.concatenate([image_true, image_fake])
    y = np.ones([2 * batch_size, 1])
    y[batch_size:, :] = 0
    d_loss = d_model.train_on_batch(x, y)

    y = np.ones([batch_size, 1])
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    d_model.trainable = False
    a_loss = ad_model.train_on_batch(noise, y)
    d_model.trainable = True

    log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
    log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
    print log_mesg

# ------------- keras adversaial model ------------------
# gan = simple_gan(model_G, model_D, normal_latent_sampling((latend_dim,)))
# model = AdversarialModel(base_model=gan, player_params=[model_G.trainable_weights, model_D.trainable_weights], player_names=['G', 'D'])
# #model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'],
# #                          loss='binary_crossentropy')
# model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=[RMSprop(lr=0.001, decay=3e-8), RMSprop(lr=0.001, decay=3e-8)],
#                           loss='binary_crossentropy')
# tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#
# num_epochs=100
# gan.summary()
# history = model.fit(x=X, y=gan_targets(X.shape[0]), epochs=num_epochs, batch_size=20, callbacks=[tbCallBack])
#

import matplotlib.pyplot as plt

# zsamples = np.random.normal(size=(10, latend_dim))
# pred = model_G.predict(zsamples).reshape(10, 28, 28)
noise = np.random.uniform(-1, 1.0, size=[10, latend_dim])
pred = model_G.predict(noise).reshape(10, 28, 28)

for i in range(pred.shape[0]):
    fig = plt.figure()
    plt.imshow(pred[i, :], cmap='gray')
    plt.savefig("pred_%d" % i)
    plt.clf()
print "pred done"

# # loss
# x=xrange(num_epochs)
# plt.plot(x, history.history['loss'])
# plt.plot(x, history.history['D_loss'])
# plt.plot(x, history.history['G_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss', 'D_loss', 'G_loss'], loc='upper left')
# plt.savefig('result_loss.png')
# plt.clf()
#
#
# plt.plot(x, history.history['G_yreal_loss'])
# plt.plot(x, history.history['G_yfake_loss'])
# plt.plot(x, history.history['D_yreal_loss'])
# plt.plot(x, history.history['D_yfake_loss'])
# plt.title('model fake real loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['G_yreal_loss', 'G_yfake_loss', 'D_yreal_loss', 'D_yfake_loss'], loc='upper left')
# plt.savefig('result_fake_real_loss.png')
