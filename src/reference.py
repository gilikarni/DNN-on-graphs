# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from diffusion_maps import create_embedding
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_uniform(shape, -1/np.sqrt(IMAGE_PIXELS), 1/np.sqrt(IMAGE_PIXELS))
    return tf.Variable(weights)


def forwardProp(X, w_1, w_2, b):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.nn.sigmoid(tf.matmul(X, w_1) + b)  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def main():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_x = mnist.train.images
    test_x = mnist.test.images
    train_y = mnist.train.labels
    test_y = mnist.test.labels
    
    if "d" in sys.argv:
        train_x = create_embedding(train_x)
        test_x = create_embedding(test_x)

    # Layer's sizes
    x_size = train_x.shape[1]   # Number of input nodes
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes 10 digits

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))
    b = tf.Variable(tf.zeros([h_size]))

    # Forward propagation
    yhat = forwardProp(X, w_1, w_2, b)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(10):
        # Train with each example
        for i in range(len(train_x)):
            sess.run(updates, feed_dict={X: train_x[i: i + 1], y: train_y[i: i + 1]})

    test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_x, y: test_y}))

    print("Test accuracy = %.2f%%" % (100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()