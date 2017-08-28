from datetime import datetime

from matplotlib.pyplot import savefig
from scipy import spatial
from numpy import linalg
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
LOG_FILE_NAME = "logs.txt"


def dist_between_vectors(vector_a, vector_b):
    return spatial.distance.euclidean(vector_a, vector_b)


def create_markov_matrix(x):
    D = [[dist_between_vectors(vector_a, vector_b) for vector_b in x] for vector_a in x]
    sigma = np.median(D)
    P = [[np.exp((-1) * (p/sigma)) for p in v] for v in D]
    for idx, row in enumerate(P):
        s = sum(row)
        P[idx] = [element/s for element in row]
    return np.array(P)


def mul(vector, num):
    return [num*x for x in vector]

# Args:
# 1. Name of input file
# 2. Name of output file
# 3. t = The length of the paths


def create_embedding():

    if len(sys.argv) < 4:
        exit("Missing arguments")

    logs = open(LOG_FILE_NAME, "a+")

    with open(sys.argv[1], 'rb') as p:
        x = pickle.load(p)

    logs.write("%s: Start creating the embedding for %s, opened the file successfully.\n" % (datetime.now(), sys.argv[1]))

    P = create_markov_matrix(x)

    logs.write("%s: Created the Markov matrix successfully.\n" % (datetime.now()))
    eigen_values, eigen_vectors = linalg.eig(P)
    plt.plot(eigen_values)
    plt.xlim(0, 1000)
    fig = plt.gcf()
    plt.show()
    fig.savefig('eigen_values Graph ' + datetime.now().strftime('%Y_%m_%d  %H_%M') + '.pdf')
    fig.savefig('eigen_values Graph ' + datetime.now().strftime('%Y_%m_%d  %H_%M') + '.jpg')
    eigen_values = eigen_values[:100]
    eigen_vectors = eigen_vectors[:100]
    embedding = []

    logs.write("%s: Calculated the eigen values successfully.\n" % (datetime.now()))

    for i in range(len(eigen_values)):
        phi = []
        for j in range(len(eigen_values)):
            phi.append(pow(eigen_values[i], int(sys.argv[3])) * eigen_vectors[i][j])
        embedding.append(phi)

    logs.write("%s: Created the Embedding matrix successfully.\n" % (datetime.now()))


    # Normalize the matrix to [-1,1]
    embedding_arr = np.array(embedding)
    maximum = np.amax(embedding_arr)
    minimum = np.amin(embedding_arr)

    embedding_arr = np.array([[((cell - minimum)*(2/(maximum - minimum)) - 1) for cell in row] for row in embedding])

    logs.write("%s: Normalized the Embedding matrix to the range [-1,1]."
               " Old maximum value = %f, old minimum value = %f. "
               "New maximum = %f, new minimum %f\n"
               % (datetime.now(), maximum, minimum, np.amax(embedding_arr), np.amin(embedding_arr)))

    with open(sys.argv[2], 'wb') as p:
        pickle.dump(embedding_arr, p, pickle.HIGHEST_PROTOCOL)

    logs.write("%s: Created the file %s with the Embedding matrix successfully.\n"
               % (datetime.now(), sys.argv[2]))

    logs.write("\n")
    logs.close()


if __name__ == '__main__':
    create_embedding()
