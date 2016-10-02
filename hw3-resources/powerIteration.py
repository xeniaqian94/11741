from scipy.sparse import csr_matrix
import numpy as np
import time


def powerIteration(alpha, M, r, p_0):
    empty_row = np.where(M.sum(axis=1) == 0)[0]  # which will be empty_column after transpose
    N = M.shape[0]
    M_transpose = M.transpose()

    M_transpose_indices = M_transpose.indices
    M_transpose_data = M_transpose.data
    M_transpose_indptr = M_transpose.indptr

    runtime = time.time()
    M_transpose_mul_r_part1 = M_transpose * r
    print "Part 1 Seconds " + str(time.time() - runtime)

    runtime = time.time()
    M_transpose_mul_r_part2 = np.ones(N) * 1.0 * sum([r[i] for i in empty_row]) / N
    print "Part 2 Seconds " + str(time.time() - runtime)

    return (1 - alpha) * (M_transpose_mul_r_part1 + M_transpose_mul_r_part2) + alpha * p_0




    # M_transpose_mul_r = np.zeros(r.shape[0], dtype=np.float64)
    #
    # runtime=time.time()
    # print runtime
    # for j in range(N):
    #     if j % 10000 == 0:
    #         print "Seconds "+str(time.time()-runtime)+" Calculating from i to document " + str(j)
    #
    #     for col_ind in range(M_transpose_indptr[j], M_transpose_indptr[j + 1]):
    #         i = M_transpose_indices[col_ind]
    #         M_transpose_mul_r[j] += M_transpose_data[col_ind] * r[i]
    #     for i in empty_row:
    #         M_transpose_mul_r[j] += 1.0 / N * r[i]

    # return (1 - alpha) * M_transpose_mul_r + alpha * p_0
