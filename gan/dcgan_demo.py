from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import argparse
import math
import pandas as pd


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2), dim_ordering="th"))
    model.add(Conv2D(64, 5, 5, border_mode='same', dim_ordering="th"))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2), dim_ordering="th"))
    model.add(Conv2D(1, 5, 5, border_mode='same', dim_ordering="th"))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, 5, 5,
                     border_mode='same',
                     input_shape=(1, 28, 28),
                     dim_ordering="th"))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Conv2D(128, 5, 5, border_mode='same', dim_ordering="th"))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[0, :, :]
    return image


def train(BATCH_SIZE):
    train = pd.read_csv('datasets/train.csv')
    train = train.drop(['label'], axis=1)
    X = train.as_matrix()
    X = X / 255.
    X_train = X.reshape(X.shape[0], 1, 28, 28)

    discriminator = discriminator_model()
    print "summary of D:"
    discriminator.summary()

    generator = generator_model()
    print "summary of G:"
    generator.summary()

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    print "summary of GAN:"
    discriminator_on_generator.summary()

    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            print image_batch.shape
            print generated_images.shape
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("iter %d, batch %d d_loss : %f" % (epoch, index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("iter %d, batch %d g_loss : %f" % (epoch, index, g_loss))
    output(generator, 100)


# def generate(BATCH_SIZE, nice=False):
#     generator = generator_model()
#     generator.compile(loss='binary_crossentropy', optimizer="SGD")
#     generator.load_weights('generator')
#     if nice:
#         discriminator = discriminator_model()
#         discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
#         discriminator.load_weights('discriminator')
#         noise = np.zeros((BATCH_SIZE*20, 100))
#         for i in range(BATCH_SIZE*20):
#             noise[i, :] = np.random.uniform(-1, 1, 100)
#         generated_images = generator.predict(noise, verbose=1)
#         d_pret = discriminator.predict(generated_images, verbose=1)
#         index = np.arange(0, BATCH_SIZE*20)
#         index.resize((BATCH_SIZE*20, 1))
#         pre_with_index = list(np.append(d_pret, index, axis=1))
#         pre_with_index.sort(key=lambda x: x[0], reverse=True)
#         nice_images = np.zeros((BATCH_SIZE, 1) +
#                                (generated_images.shape[2:]), dtype=np.float32)
#         for i in range(int(BATCH_SIZE)):
#             idx = int(pre_with_index[i][1])
#             nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
#         image = combine_images(nice_images)
#     else:
#         noise = np.zeros((BATCH_SIZE, 100))
#         for i in range(BATCH_SIZE):
#             noise[i, :] = np.random.uniform(-1, 1, 100)
#         generated_images = generator.predict(noise, verbose=1)
#         image = combine_images(generated_images)
#     image = image*127.5+127.5
#     Image.fromarray(image.astype(np.uint8)).save(
#         "generated_image.png")

def output(generator, latend_dim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    noise = np.random.uniform(-1, 1.0, size=[10, latend_dim])
    pred = generator.predict(noise).reshape(10, 28, 28)

    for i in range(pred.shape[0]):
        # fig = plt.figure()
        plt.imshow(pred[i, :], cmap='gray')
        plt.savefig("pred_dcgan_%d" % i)
        plt.clf()
    print "pred done"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train(64)
