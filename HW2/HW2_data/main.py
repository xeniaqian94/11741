import load_docVec
import randomInitialization
import sys

import scipy.sparse as sps

from sklearn.preprocessing import normalize
import numpy as np

import assignment
import update

n, d, M = load_docVec.load_docVec(sys.argv[1])

iter = 10
tol = 1e-4
k = 20

centroids = randomInitialization.randomInitialization(n, d, M, k)
similarity = np.array(np.zeros(n), dtype=np.float64)
labels = -np.ones(n)
for i in range(iter):
    print "Within " + str(i + 1) + " iteration!"
    old_centroids = centroids.copy()
    old_similarity = similarity.copy()
    old_labels = labels.copy()
    labels, similarity = assignment.assignment(M, centroids)
    print np.transpose(labels)
    centroids = update.update(M, labels, k)
    if (abs(similarity.sum() - old_similarity.sum()) < tol) \
            or ((labels == old_labels).all()):
        print "Converged after " + str(i + 1) + " turns"
        break
    else:
        print "New similarity " + str(similarity.sum()) + " Old similarity " + str(old_similarity.sum())

f = open("HW2_dev.eval_output", 'w')

for i in range(n):
    # print str(i) + " " + str(labels[i])
    f.write(str(i) + " " + str(labels[i])+"\n")
f.close()

# M_normalized = csr_matrix([[1, 12, 0], [0, 0, 3], [4, 0, 5]])
# # print A[0,1]
#
# M_indices = M_normalized.indices
# M_data = M_normalized.data
# M_indptr = M_normalized.indptr
#
# print M_indices
# print M_data[4]
# print M_indptr  # # print (sps.issparse(A.dot(np.asarray([1, 2, 3]))))
#
# B = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
# for i in range(3):
#     dist = B.dot(np.transpose(A[i].toarray()))
#     print dist
#     # print np.argmax(dist)
