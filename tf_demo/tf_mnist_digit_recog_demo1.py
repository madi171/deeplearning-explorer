# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

mnist = input_data.read_data_sets("datasets/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(10000):
    bs_xs, bs_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: bs_xs, y_: bs_ys})
    print "\tbatch loss=%f" % sess.run(cross_entropy, feed_dict={x: bs_xs, y_: bs_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
