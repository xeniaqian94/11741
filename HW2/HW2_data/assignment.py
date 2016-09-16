import numpy as np
from sklearn.preprocessing import normalize


# Assignment step
def assignment(M, centroids):
    n = M.shape[0]
    d = M.shape[1]
    k = centroids.shape[0]

    M_normalized = normalize(M, norm="l2", axis=1)
    M_indices = M_normalized.indices
    M_data = M_normalized.data
    M_indptr = M_normalized.indptr

    centroids_normalized = normalize(centroids, norm="l2", axis=1)
    centroids_normalized_transpose = np.transpose(centroids_normalized)

    labels = -np.ones(n,dtype=np.int64)
    similarity = np.array(np.zeros(n), dtype=np.float64)

    for i in range(n):
        max_similarity = -1
        for j in range(k):
            this_similarity = 0
            for l in range(M_indptr[i], M_indptr[i + 1]):
                this_similarity += centroids_normalized[j, M_indices[l]] * M_data[l]
            if (max_similarity < this_similarity):
                max_similarity = this_similarity
                labels[i] = j

        similarity[i] = max_similarity

    return labels, similarity
