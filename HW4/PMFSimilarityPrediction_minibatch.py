import csv
from scipy.sparse import csr_matrix
import numpy as np
import time
import pickle

ratingDict = dict()
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        ratingDict[(int(row[0]), int(row[1]))] = int(row[2])  # cold start imputation

dimension = 5
learning_rate = 0.01
threshold = 0.1
sigma_square = 1
sigma_square_u = 10
sigma_square_v = 100
lambda_u = float(sigma_square) / sigma_square_u
lambda_v = float(sigma_square) / sigma_square_v

currentTime = time.time()


def sparse_user_item_matrix(ratingDict):
    row = list()
    col = list()
    data = list()

    for movie_user_pair in ratingDict:
        row.append(movie_user_pair[1])
        col.append(movie_user_pair[0])
        data.append(ratingDict[movie_user_pair])

    return csr_matrix((np.asarray(data), (np.asarray(row), np.asarray(col))), dtype=np.float64)


M = sparse_user_item_matrix(ratingDict)  # num_user * num_item
I = M.copy()
print len(M.data)
# print M.todense()[1, 955]

I.data /= I.data

num_user = M.shape[0]
num_item = M.shape[1]

U = np.random.random((dimension, num_user))
V = np.random.random((dimension, num_item))

converged = False

currentTime=time.time()

# f2 = open('U_' + str(dimension) + "_SGD.txt", 'r')
# U = pickle.load(f2)
# f2.close()
#
# f2 = open('V_' + str(dimension) + "_SGD.txt", 'r')
# V = pickle.load(f2)
# f2.close()

new_U = np.random.random((dimension, num_user))
new_V = np.random.random((dimension, num_item))

def likelihood(M, new_U, new_V):  # This is the objective function to be minimized
    error_matrix = M - csr_matrix(new_U.transpose().dot(new_V))
    error_matrix.data **= 2

    quadratic_U = np.sum(np.asarray(new_U) ** 2)
    quadratic_V = np.sum(np.asarray(new_V) ** 2)

    print "quadratic_U/V " + str(quadratic_U) + " " + str(quadratic_V)

    return 0.5 * np.sum(
        (error_matrix.multiply(I)).todense()) + 0.5 * lambda_u * quadratic_U + 0.5 * lambda_v * quadratic_V


def update(M, U, V, learning_rate, threshold, converged, keys):
    print "Time passed "+str(time.time() - currentTime)
    gradient_U = np.zeros((dimension, num_user))
    gradient_V = np.zeros((dimension, num_item))

    r_hat = U.transpose().dot(V)

    for key in keys:
        movieID = key[0]
        userID = key[1]
        rating = ratingDict[key]
        for d in range(dimension):
            gradient_U[d, userID] -= V[d, movieID] * (rating - r_hat[userID, movieID])
            gradient_V[d, movieID] -= U[d, userID] * (rating - r_hat[userID, movieID])

    # print gradient_U

    gradient_U += lambda_u * U
    gradient_V += lambda_v * V

    # print gradient_U

    while (not converged):
        last_likelihood = likelihood(M, U, V)
        print "last likelihhod is " + str(last_likelihood)
        # print gradient_U
        # print gradient_V

        new_U = U - learning_rate * gradient_U
        # print U[0, 0], new_U[0, 0]
        new_V = V - learning_rate * gradient_V

        new_likelihood = likelihood(M, new_U, new_V)
        print "new likelihhod is " + str(new_likelihood)

        if new_likelihood < last_likelihood:
            print "Good trend converging, accept this update "
            learning_rate *= 1.25
            U = new_U
            V = new_V

            if last_likelihood - new_likelihood < threshold :
                converged = True
            f_write = open(
                "eval/" + str(dimension) + "_" + str(sigma_square) + "_" + str(sigma_square_u) + "_" + str(
                    sigma_square_v) + "SGD.txt", "w")
            r_hat = U.transpose().dot(V)
            with open('dev.csv', 'rb') as f_dev:
                reader = csv.reader(f_dev)
                for row in reader:
                    movieID = int(row[0])
                    userID = int(row[1])
                    f_write.write(str(r_hat[userID, movieID]) + "\n")
            f_write.close()

            f = open('U_' + str(dimension) + "_SGD.txt", 'w')
            pickle.dump(U, f)
            f.close()

            f = open('V_' + str(dimension) + "_SGD.txt", 'w')
            pickle.dump(V, f)
            f.close()
            break
        else:
            learning_rate *= 0.5
            print "Bad trend, not convergine!!!!, reject this update " + str(learning_rate)
            break

        if learning_rate < 1e-10:
            converged = True
    return M, U, V, learning_rate, threshold, converged


key = ratingDict.keys()[0:10000]
[M, U, V, learning_rate, threshold, converged] = update(M, U, V, learning_rate, threshold, converged, key)
i = 10000

instance_size = len(ratingDict.keys())

while not converged:
    print i%10000
    if i > instance_size:
        i = 0
    key = ratingDict.keys()[i:i+10000]
    # print key
    [M, U, V, learning_rate, threshold, converged] = update(M, U, V, learning_rate, threshold, converged, key)
    i += 10000
    print time.time()-currentTime

