import load_docVec
import sys
import numpy as np
import assignment
import update
import kMeansPlusPlus_Initialization

n, d, M = load_docVec.load_docVec(sys.argv[1])
iter = 1000
tol = 1e-4
n_cluster = [67]


for k in n_cluster:
    for time in range(10):

        centroids = kMeansPlusPlus_Initialization.kMeansPlusPlus_Initialization(n, d, M, k)
        # centroids=random_Initialization.randomInitialization(n,d,M,k)
        similarity = np.array(np.zeros(n), dtype=np.float64)

        labels = -np.ones(n)
        for i in range(iter):
            print "Within " + str(i + 1) + " iteration!"
            old_centroids = centroids.copy()
            old_similarity = similarity.copy()
            old_labels = labels.copy()
            labels, similarity = assignment.assignment(M, centroids)
            # print np.transpose(labels)
            centroids = update.update(M, labels, k)
            if (abs(similarity.sum() - old_similarity.sum()) < tol) \
                    or ((labels == old_labels).all()):
                print "Converged after " + str(i + 1) + " turns"
                break
            else:
                print "New similarity " + str(similarity.sum()) + " Old similarity " + str(old_similarity.sum())

        f = open("HW2_dev.eval_output_" + str(k) + "_" + str(time) + "_kmeans++",
                 'w')  # f = open("HW2_dev.eval_output_" + str(k) + "_" + str(time)+"_tuneBaslinek", 'w')

        for i in range(n):
            f.write(str(i) + " " + str(labels[i]) + "\n")
        f.close()
