import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from diffusion_maps import create_embedding
import sys
import ssl
import pickle


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_x = mnist.train.images
    test_x = mnist.test.images
    train_y = mnist.train.labels
    test_y = mnist.test.labels
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