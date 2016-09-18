from sklearn.preprocessing import normalize
import numpy as np
from scipy.sparse import csr_matrix

# A = np.zeros(10)
# A[2] = 3
#
# print A.sum()
#
# tally = np.bincount([0, 1, 4, 2, 3, 9], minlength=10)
# empty_clusters = np.where(tally == 0)[0]
# print empty_clusters.shape[0]
# print 3 / tally[1]
#
# M = np.matrix([[1, 2, 3], [4, 5, 6], [0, 2, 2]])
# M_normalized = normalize(M, norm="l2", axis=1)
# print M_normalized
#
# similarity = np.zeros(2)
# similarity[0] = 5.0
# print similarity
#
# M = csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# A = np.zeros((3, 3))
# A[0] += M[0]
#
# similarity = np.array(np.zeros(10), dtype=np.float64)
#
# print similarity.sum()
# print range(10)
#
# centroids = np.zeros((10, 3))
# centroids[1,2]=3
# centroids[1,1]=2
# centroids_normalized = normalize(centroids, norm="l2", axis=1)
# print centroids_normalized

distance = np.array(np.zeros(5), dtype=np.float64)
distance=[1.1,2.2,3.3,4.4,5.5]
print range(10)

print np.random.choice(np.arange(1, 7), p=[0.46, 0.01, 0.01, 0.01, 0.01, 0.5])