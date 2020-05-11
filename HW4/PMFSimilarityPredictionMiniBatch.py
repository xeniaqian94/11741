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

dimension = 10
learning_rate = 0.0001
threshold = 100
sigma_square = 1
sigma_square_u = 100
sigma_square_v = 1000
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
print len(M.data)
I = M.copy()
print M.todense()[1, 955]

I.data /= I.data

num_user = M.shape[0]
num_item = M.shape[1]

U = np.random.random((dimension, num_user))
V = np.random.random((dimension, num_item))

converged = False

new_U = np.random.random((dimension, num_user))
new_V = np.random.random((dimension, num_item))


def likelihood(M, new_U, new_V):  # This is the objective function to be minimized
    error_matrix = M - csr_matrix(new_U.transpose().dot(new_V))
    error_matrix.data **= 2

    # print np.linalg.norm(new_U, axis=0).shape
    # quadratic_U = np.sum(np.linalg.norm(new_U, axis=0))
    # quadratic_U = np.sum(np.apply_along_axis(np.linalg.norm, 0, new_U))
    quadratic_U=np.sum(np.asarray(new_U) ** 2)
    # print np.linalg.norm(new_V, axis=0).shape
    # quadratic_V = np.sum(np.linalg.norm(new_V, axis=0))
    # quadratic_V = np.sum(np.apply_along_axis(np.linalg.norm, 0, new_V))
    quadratic_V=np.sum(np.asarray(new_V) ** 2)

    print "quadratic_U/V " + str(quadratic_U) + " " + str(quadratic_V)

    return 0.5 * np.sum(
        (error_matrix.multiply(I)).todense()) + 0.5 * lambda_u * quadratic_U + 0.5 * lambda_v * quadratic_V


def update(M, U, V, learning_rate, threshold, converged,i,j):
    print dimension, sigma_square_u, sigma_square_v, lambda_u, lambda_v
    print "Already running for seconds " + str(time.time() - currentTime)
    gradient_U = np.zeros((dimension, num_user))
    gradient_V = np.zeros((dimension, num_item))

    r_hat = csr_matrix(U.transpose().dot(V))
    loss_matrix = (M - r_hat).multiply(I)
    gradient_U = -V * loss_matrix.transpose() + lambda_u * U
    gradient_V = -U * loss_matrix + lambda_v * V

    print "gradient two parts"
    print -V * loss_matrix.transpose()
    print lambda_u * U



    # gradient_U = -V * loss_matrix.transpose() + 0.5 * lambda_u * np.divide(U, np.apply_along_axis(np.linalg.norm, 0, U))
    # gradient_U = -V * loss_matrix.transpose() + 0.5 * lambda_u * np.divide(U, np.linalg.norm(U, axis=0))
    # gradient_V = -U * loss_matrix + 0.5 * lambda_v * np.divide(V, np.linalg.norm(V, axis=0))
    # gradient_V = -U * loss_matrix + 0.5 * lambda_v * np.divide(V, np.apply_along_axis(np.linalg.norm, 0, V))

    while (not converged):
        last_likelihood = likelihood(M, U, V)
        print "last likelihhod is " + str(last_likelihood)
        # print gradient_U
        # print gradient_V

        new_U = U - learning_rate * gradient_U

        # print U, gradient_U
        # print U[0, 0], new_U[0, 0]
        new_V = V - learning_rate * gradient_V

        new_likelihood = likelihood(M, new_U, new_V)
        print "new likelihhod is " + str(new_likelihood)



        if new_likelihood < last_likelihood:
            print "Good trend converging, accept this update "
            learning_rate *= 1.25
            U = new_U
            V = new_V

            if last_likelihood - new_likelihood < threshold and new_likelihood < 390000:
                converged = True
            # if new_likelihood < 418000:
            f_write = open(
                "eval/" + str(dimension) + "_" + str(sigma_square) + "_" + str(sigma_square_u) + "_" + str(
                    sigma_square_v) + ".txt", "w")
            r_hat = U.transpose().dot(V)
            with open('dev.csv', 'rb') as f_dev:
                reader = csv.reader(f_dev)
                for row in reader:
                    movieID = int(row[0])
                    userID = int(row[1])
                    f_write.write(str(r_hat[userID, movieID]) + "\n")
            f_write.close()
            break
        else:
            learning_rate *= 0.5
            print "!!!! updating learning rate" + str(learning_rate)
            break

        if learning_rate < 1e-10:
            converged = True

    return M, U, V, learning_rate, threshold, converged

    # for d in [5, 10, 20, 50]:
    #     for j in [10, 100]:
    #         for k in [100, 1000]:

for d in [10]:
    for j in [100]:
        for k in [1000]:
            currentTime = time.time()
            dimension = d
            sigma_square_u = j
            sigma_square_v = k

            lambda_u = float(sigma_square) / sigma_square_u
            lambda_v = float(sigma_square) / sigma_square_v
            converged = False

            [M, U, V, learning_rate, threshold, converged] = update(M, U, V, learning_rate, threshold, converged)
            i = 1

            while not converged:
                print i
                [M, U, V, learning_rate, threshold, converged] = update(M, U, V, learning_rate, threshold, converged)
                i += 1
            print time.time() - currentTime

            f_write = open(
                "eval/" + str(dimension) + "_" + str(sigma_square) + "_" + str(sigma_square_u) + "_" + str(
                    sigma_square_v) + ".txt", "w")
            r_hat = U.transpose().dot(V)
            with open('dev.csv', 'rb') as f_dev:
                reader = csv.reader(f_dev)
                for row in reader:
                    movieID = int(row[0])
                    userID = int(row[1])
                    f_write.write(str(r_hat[userID, movieID]) + "\n")
            f_write.close()

            f = open('U_' + str(dimension) + "_" + str(sigma_square) + "_" + str(sigma_square_u) + "_" + str(
                sigma_square_v) + ".txt", 'w')
            pickle.dump(U, f)
            f.close()

            f = open('V_' + str(dimension) + "_" + str(sigma_square) + "_" + str(sigma_square_u) + "_" + str(
                sigma_square_v) + ".txt", 'w')
            pickle.dump(V, f)
            f.close()

            # f2 = open('test', 'r')
            # s = pickle.load(f2)
            # f2.close()
