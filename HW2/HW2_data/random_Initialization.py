import random
import numpy as np


def randomInitialization(n, d, M, k):
    print "Document size " + str(n) + " Dictionary size " + str(d)

    initial_seed_index = random.sample(xrange(n), k)
    print "Initial seed index " + str(initial_seed_index)

    centroids = np.empty((k, d))

    for i in range(0, k):
        centroids[i] = M[initial_seed_index[i]].toarray()
    return centroids
