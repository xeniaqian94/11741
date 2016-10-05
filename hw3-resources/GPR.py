from scipy.sparse import csr_matrix
import numpy as np
import sys
import readTransitionMatrix
import scipy.stats as ss
import powerIteration

import time

currentTime=time.time()

print sys.argv[1]
alpha = 0.8
beta = 0
gamma = 0.2
iter = 500

M = readTransitionMatrix.readTransitionMatrix(sys.argv[1])

r = np.zeros(M.shape[0])
r[0] = 1

p_0 = np.ones(M.shape[0])
p_0 = p_0 / M.shape[0]
p_t = p_0  # could be ignored in this GPR implementation

vectorValue = file(sys.argv[2], "w")

for i in range(iter):
    print "Within " + str(i + 1) + " iteration!"
    rank_old = ss.rankdata(r, method='ordinal')

    r_new = powerIteration.powerIteration(alpha, beta, gamma, M, r, p_0, p_t)
    rank_new = ss.rankdata(r_new, method='ordinal')
    print "rank old " + str(M.shape[0] - 1 - rank_old.astype(int))
    print "rank new " + str(M.shape[0] - 1 - rank_new.astype(int))
    print rank_new - rank_old

    if ((rank_new - rank_old) == 0).all():
        print "Converged after " + str(i + 1) + " turns"
        print r - r_new
        r = r_new
        break
    else:
        print "New r as "
        print r_new
        r = r_new

print r



for i in range(len(r)):
    vectorValue.write(str(i + 1) + " " + str(r[i]) + "\n")

print "Finished in seconds "+str(time.time()-currentTime)
