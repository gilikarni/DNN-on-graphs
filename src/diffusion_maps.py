from scipy import spatial
from numpy import linalg

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def dist_between_vectors(vector_a, vector_b):
    return spatial.distance.euclidean(vector_a, vector_b)


def create_markov_matrix(x):
    P = []
    for vector_a in x:
        dist = []
        for vector_b in x:
            dist.append(dist_between_vectors(vector_a, vector_b))
        sum_of_dist = sum(dist)
        dist = [x/sum_of_dist for x in dist]
        P.append(dist)
    return P


def mul(vector, num):
    return [num*x for x in vector]


def create_embedding(x):
    P = create_markov_matrix(x)
    eigen_values, eigen_vectors = linalg.eig(P)
    embedding = []
    for i in range(len(eigen_values)):
        embedding.append(mul(eigen_vectors[i], eigen_values[i]))
    return embedding



