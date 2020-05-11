import csv
from scipy.sparse import csr_matrix
import numpy as np
import time
import pickle

for d in [2, 5, 10, 20]:
    for j in [10]:
        for k in [100]:
            currentTime = time.time()
            dimension = d
            sigma_square=1
            sigma_square_u = j
            sigma_square_v = k

            f2 = open('U_' + str(dimension) + "_batch_intermediate.txt", 'r')
            U = pickle.load(f2)
            f2.close()

            f2 = open('V_' + str(dimension) + "_batch_intermediate.txt", 'r')
            V = pickle.load(f2)
            f2.close()

            f_write = open("eval/" + str(dimension) + "_" + str(sigma_square) + "_" + str(sigma_square_u) + "_" + str(
                sigma_square_v) + "_predictions.txt", "w")
            r_hat = U.transpose().dot(V)
            with open('test.csv', 'rb') as f_dev:
                reader = csv.reader(f_dev)
                for row in reader:
                    movieID = int(row[0])
                    userID = int(row[1])
                    f_write.write(str(r_hat[userID, movieID]) + "\n")
            f_write.close()
            f_write = open("eval/" + str(dimension) + "_" + str(sigma_square) + "_" + str(sigma_square_u) + "_" + str(
                sigma_square_v) + "_validation.txt", "w")
            r_hat = U.transpose().dot(V)
            with open('dev.csv', 'rb') as f_dev:
                reader = csv.reader(f_dev)
                for row in reader:
                    movieID = int(row[0])
                    userID = int(row[1])
                    f_write.write(str(r_hat[userID, movieID]) + "\n")
            f_write.close()
