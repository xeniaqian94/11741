import random
import numpy as np
from sklearn.preprocessing import normalize


def kMeansPlusPlus_Initialization(n, d, M, k):
    M_normalized = normalize(M, norm="l2", axis=1)
    M_indices = M_normalized.indices
    M_data = M_normalized.data
    M_indptr = M_normalized.indptr

    print "Document size " + str(n) + " Dictionary size " + str(d)

    initial_seed_index = random.randrange(0, n)
    print "Initial seed index " + str(initial_seed_index)

    centroids = np.zeros((k, d))
    centroids[0] = M[initial_seed_index].toarray()
    centroids_normalized = normalize(centroids, norm="l2", axis=1)

    # choose the next center
    for i in range(1, k):
        distance = np.array(np.zeros(n), dtype=np.float64)
        for j in range(n):
            max_similarity = -1
            # Find the closest distance
            for l in range(i):
                this_similarity = 0
                for m in range(M_indptr[j], M_indptr[j + 1]):
                    this_similarity += centroids_normalized[l, M_indices[m]] * M_data[m]
                if (max_similarity < this_similarity):
                    max_similarity = this_similarity
            if max_similarity > 1:
                distance[j] = 0
            else:
                distance[j] = 1 - max_similarity

        distance_norm = distance / np.linalg.norm(distance, ord=1)
        candidate_seed_index = np.random.choice(range(n), p=distance_norm)
        print distance_norm[initial_seed_index]
        print "chosed centroid " + str(i) + " as " + str(candidate_seed_index)
        centroids[i] = M[candidate_seed_index].toarray()
        centroids_normalized = normalize(centroids, norm="l2", axis=1)

    return centroids
