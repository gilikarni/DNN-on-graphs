from datetime import datetime

from scipy import spatial
from numpy import linalg
import sys
import pickle
import numpy as np

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
LOG_FILE_NAME = "logs.txt"


def dist_between_vectors(vector_a, vector_b):
    return np.exp(-1 * (spatial.distance.euclidean(vector_a, vector_b)))


def create_markov_matrix(x):
    P = []
    i = 0
    for vector_a in x:
        dist = []
        if 0 == (i % 100):
            print("i = %d" % i)
        i += 1
        for vector_b in x:
            dist.append(dist_between_vectors(vector_a, vector_b))
        sum_of_dist = sum(dist)
        dist = [x/sum_of_dist for x in dist]
        P.append(dist)
    return np.array(P)


def mul(vector, num):
    return [num*x for x in vector]

# Args:
# 1. Name of input file
# 2. Name of output file


def create_embedding():

    if len(sys.argv) < 3:
        exit("Missing arguments")

    logs = open(LOG_FILE_NAME, "a+")

    with open(sys.argv[1], 'rb') as p:
        x = pickle.load(p)

    logs.write("%s: Start creating the embadding for %s, opened the file successfully.\n" % (datetime.now(), sys.argv[1]))

    P = create_markov_matrix(x)

    logs.write("%s: Created the Markov matrix successfully.\n" % (datetime.now()))
    eigen_values, eigen_vectors = linalg.eig(P)
    embedding = []

    logs.write("%s: Calculated the eigen values successfully.\n" % (datetime.now()))

    for i in range(len(eigen_values)):
        embedding.append(mul(eigen_vectors[i], eigen_values[i]))

    logs.write("%s: Created the Embedding matrix successfully.\n" % (datetime.now()))

    with open(sys.argv[2], 'wb') as p:
        pickle.dump(np.array(embedding), p, pickle.HIGHEST_PROTOCOL)

    logs.write("%s: Created the file %s with the Embedding matrix successfully.\n" % (datetime.now(), sys.argv[2]))

    logs.close()


if __name__ == '__main__':
    create_embedding()
