import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from diffusion_maps import create_embedding
import sys
import ssl
import pickle

train_size = 10000
test_size = 2000


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_x = np.delete(mnist.train.images, np.s_[train_size:], 0)
    test_x = np.delete(mnist.test.images, np.s_[test_size:], 0)
    train_y = np.delete(mnist.train.labels, np.s_[train_size:], 0)
    test_y = np.delete(mnist.test.labels, np.s_[test_size:], 0)
    with open('train_x.pkl', 'wb') as p:
        pickle.dump(train_x, p, pickle.HIGHEST_PROTOCOL)
    with open('train_y.pkl', 'wb') as p:
        pickle.dump(train_y, p, pickle.HIGHEST_PROTOCOL)
    with open('test_x.pkl', 'wb') as p:
        pickle.dump(test_x, p, pickle.HIGHEST_PROTOCOL)
    with open('test_y.pkl', 'wb') as p:
        pickle.dump(test_y, p, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()