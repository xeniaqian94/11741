from scipy.sparse import csr_matrix, lil_matrix
import numpy as np


# the column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
def readTransitionMatrix(file):
    lines = open(file, "r")

    row_sum = dict()

    for line in lines:
        entry = line.strip(" ").split(" ")
        if ((int(entry[0]) - 1) not in row_sum):
            row_sum[int(entry[0]) - 1] = 1
        else:
            row_sum[int(entry[0]) - 1] += int(entry[2])

    lines = open(file, "r")
    row = list()
    col = list()
    data = list()

    for line in lines:
        entry = line.strip(" ").split(" ")
        row.append(int(entry[0]) - 1)
        col.append(int(entry[1]) - 1)
        data.append(1.0 * int(entry[2]) / row_sum[int(entry[0]) - 1])

    print len(row), len(col), len(data)
    M = csr_matrix((np.asarray(data), (np.asarray(row), np.asarray(col))), dtype=np.float64)
    print "Matrix has documents " + str(M.shape[0])+ " "+str(M.shape[1])

    M = csr_matrix((np.asarray(data), (np.asarray(row), np.asarray(col))), shape=[M.shape[0], M.shape[0]],
                   dtype=np.float64)

    print "Empty rows are " + str(np.where(M.sum(axis=1) == 0)[0])
    print "Matrix has documents " +str(M.shape[0])+ " "+str(M.shape[1])

    return M
