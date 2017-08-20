from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import time

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = {}
FLAGS['log_dir'] = "/home/madi/remote_python/tf_logs"


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Our application logic will be added here
def cnn_demo():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # Input Layer
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First layers cnn
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second layers cnn
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Thrid layers, full connection
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Forth layers, softmax output
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('cross_entropy_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    correct_prediction = tf.equal(tf.argmax(y_conv, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # with tf.variable_scope('conv1') as scope:
    # tf.get_variable_scope().reuse_variables()
    # W_conv1 is filters, now I want to visualize it
    # filter_summary_conv1 = tf.summary.image("filters", W_conv1, max_outputs=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS['log_dir'], sess.graph)

        for i in range(50000):
            time.sleep(0.005)
            batch = mnist.train.next_batch(100)
            if i % 2000 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                train_op.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.4})

                # record summary info to disk
                summary = sess.run([merged, accuracy], feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print(summary)
                train_writer.add_summary(summary, i)
                train_writer.add
        train_writer.close()
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def main(unused_argv):
    # Load training and eval data
    if tf.gfile.Exists(FLAGS['log_dir']):
        tf.gfile.DeleteRecursively(FLAGS['log_dir'])
    cnn_demo()


if __name__ == "__main__":
    tf.app.run()
