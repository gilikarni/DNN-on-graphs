# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
import sys
import ssl
import pickle
import time
from datetime import datetime

ssl._create_default_https_context = ssl._create_unverified_context

levels = 2
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

LOG_FILE_NAME = "logs.txt"


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_uniform(shape, -1/np.sqrt(IMAGE_PIXELS), 1/np.sqrt(IMAGE_PIXELS))
    return tf.Variable(weights)


def forwardProp(X, w, b):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = X
    for i in range(len(b) - 1):
        h = tf.nn.sigmoid(tf.matmul(h, w[i]) + b[i]) # The \sigma function

    yhat = tf.matmul(h, w[-1]) + b[-1]  # The \varphi function
    return yhat

# Params:
# 1. train_x
# 2. train_y
# 3. test_x
# 4. test_y

def main():

    if len(sys.argv) < 3:
        exit("Missing arguments -\n1. train_x\n2. train_y")
    with open(sys.argv[1], 'rb') as p:
        x = pickle.load(p)
    with open(sys.argv[2], 'rb') as p:
        y = pickle.load(p)
    len_x = len(x)
    matrixesx = np.split(x, [int(0.8*len_x)])
    matrixesy = np.split(y, [int(0.8*len_x)])
    train_x = matrixesx[0]
    test_x = matrixesx[1]
    train_y = matrixesy[0]
    test_y = matrixesy[1]

    #with open(sys.argv[3], 'rb') as p:
    #test_x = pickle.load(p)
    #with open(sys.argv[4], 'rb') as p:
    #   test_y = pickle.load(p)

    logs = open(LOG_FILE_NAME, "a+")

    # Layer's sizes
    x_size = train_x.shape[1]   # Number of input nodes
    h_size = 256                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes 10 digits
    sizes = [x_size, h_size, h_size, y_size]

    logs.write("%s: Start session with hidden layer of %d neurons.\n" % (datetime.now(), h_size))

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w = []
    b = []
    for i in range(levels+1):
        w_1 = init_weights((sizes[i], sizes[i+1]))
        w.append(w_1)
        b_1 = tf.Variable(tf.zeros([sizes[i+1]]))
        b.append(b_1)

    # Forward propagation
    yhat = forwardProp(X, w, b)
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

    logs.write("%s: Test accuracy = %.2f%%\n" % (datetime.now(), 100. * test_accuracy))

    sess.close()

    logs.write("\n")
    logs.close()

if __name__ == '__main__':
    main()
