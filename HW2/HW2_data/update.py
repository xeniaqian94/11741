import random
import numpy as np


def update(M, labels, k):
    n = M.shape[0]
    d = M.shape[1]
    M_data = M.data
    M_indices = M.indices
    M_indptr = M.indptr
    centroids = np.zeros((k, d))

    # Before the update step, first handle those empty cluster, randomly pick a document to this cluster

    tally = np.bincount(labels, minlength=k)
    empty_clusters = np.where(tally == 0)[0]
    replace_seed = random.sample(xrange(n), empty_clusters.shape[0])

    print "Before empty cluster handling, tally equals " + str(tally.sum())

    for i in range(empty_clusters.shape[0]):
        tally[labels[replace_seed[i]]] -= 1
        labels[replace_seed[i]] = empty_clusters[i]
        tally[empty_clusters[i]] = 1

    print "After empty cluster handling, tally equals " + str(tally.sum())

    for i in range(n):
        centroids[labels[i]] += M[i]

    for i in range(k):
        centroids[i]/=tally[i]

    return centroids
