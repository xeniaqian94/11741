import csv
from scipy.sparse import csr_matrix
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import time
import pickle
# from loadTraining import users_by_movieID

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.preprocessing import normalize

ratingDict = dict()
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        ratingDict[(int(row[0]), int(row[1]))] = int(row[2]) - 3  # code start imputation


def users_by_movieID(movieID):
    return [ratingDict[movie_user_pair] for movie_user_pair in ratingDict.keys() if movie_user_pair[0] == movieID]


def sparse_user_item_matrix(ratingDict):
    row = list()
    col = list()
    data = list()

    for movie_user_pair in ratingDict:
        row.append(movie_user_pair[1])
        col.append(movie_user_pair[0])
        data.append(ratingDict[movie_user_pair])

    return csr_matrix((np.asarray(data), (np.asarray(row), np.asarray(col))), dtype=np.float64)


M = sparse_user_item_matrix(ratingDict)

movie_avg = np.asarray(np.sum(M.todense(), axis=0, dtype=float) / M.todense().shape[0])[0]

print M.shape

currentTime = time.time()

user_similarity_dot_product = M.dot(M.transpose())

user_mean = M.mean(1)

user_sum = M.sum(1)
print user_mean[1]
print user_sum[1]
user_norm = np.asmatrix(np.std(np.asarray(M.todense()), axis=1) * np.sqrt(M.shape[1])).transpose()
print user_norm[1]

avg_norm = user_norm.sum() / user_norm.shape[0]

for i in range(user_norm.shape[0]):
    if user_norm[i, 0] == 0:  # cold start users
        # print i
        user_norm[i, 0] = avg_norm

print time.time() - currentTime

user_similarity_dot_product = user_similarity_dot_product.todense()

user_mean_by_user_sum = np.dot(user_mean, user_sum.transpose())
user_mean_by_user_mean = np.dot(user_mean, user_mean.transpose())
user_norm_by_user_norm = np.dot(user_norm, user_norm.transpose())
user_similarity_normalized = user_similarity_dot_product - user_mean_by_user_sum - user_mean_by_user_sum.transpose() + user_mean_by_user_mean
user_similarity_normalized = np.divide(user_similarity_normalized, user_norm_by_user_norm)

print "user similarity: " + str(user_similarity_normalized[8497, 4321])

c = M.todense() - M.mean(1)
print M.mean(1)
# centering to 0 step from course slide
M_scaled = normalize(c, norm='l2', axis=1)  # normalizating step from course slide

user_similarity_normalized = np.dot(M_scaled, M_scaled.transpose())
print M_scaled[8497].mean(), np.var(M_scaled[8497])
print M_scaled[4321].mean(), np.var(M_scaled[8497])
# print np.linalg.norm(M_scaled, ord=2, axis=1)
print "dot product: " + str(np.dot(M_scaled[8497], M_scaled[4321]))
print "similarity: " + str(user_similarity_normalized[8497, 4321]) + " " + str(user_similarity_normalized[4321, 8497])

print time.time() - currentTime

user_similarity_dot_product = user_similarity_normalized - (1 - user_similarity_normalized.min()) * np.diag(
    np.diag(user_similarity_normalized))
user_similarity_cosine_similarity = user_similarity_dot_product

print user_similarity_cosine_similarity.shape


def getKNN(userID, row, k, method):
    user_weight_list = sorted(rowToDict(row).iteritems(), key=itemgetter(1), reverse=True)[:k]

    if method == "mean":
        kNN_list = [(user_weight[0], 1.0 / k) for user_weight in user_weight_list]
        return kNN_list
    elif method == "weighted_mean":
        weight_sum = sum([user_weight[1] for user_weight in user_weight_list])
        if weight_sum == 0:
            return []
        kNN_list = [(user_weight[0], user_weight[1] / weight_sum) for user_weight in user_weight_list]
        return kNN_list


def rowToDict(row):
    d = dict()
    for i in range(len(row)):
        d[i] = row[i]
    return d


print "Testing"
print getKNN(4321, np.squeeze(np.asarray(user_similarity_cosine_similarity[4321])), 5, "weighted_mean")
print getKNN(4321, np.squeeze(np.asarray(user_similarity_cosine_similarity[4321])), 5, "mean")
print getKNN(4321, np.squeeze(np.asarray(user_similarity_dot_product[4321])), 5, "weighted_mean")
print getKNN(4321, np.squeeze(np.asarray(user_similarity_dot_product[4321])), 5, "mean")
print "\n"

variants = [
    ("mean", "cosine_similarity", 10), ("mean", "cosine_similarity", 100), ("mean", "cosine_similarity", 500),
    ("weighted_mean", "cosine_similarity", 10), ("weighted_mean", "cosine_similarity", 100),
    ("weighted_mean", "cosine_similarity", 500)]

# variants = [("weighted_mean", "cosine_similarity", 10), ("weighted_mean", "cosine_similarity", 100),
#             ("weighted_mean", "cosine_similarity", 500)]

for variant in variants:

    k = int(variant[2])
    similarity_method = variant[1]  # "dot_product"
    rating_method = variant[0]  # "mean"
    print "Within this variant " + str(k) + " " + similarity_method + " " + rating_method
    f = open("eval/Experiment3_" + rating_method + "_" + similarity_method + "_" + str(k), 'w')

    currentTime = time.time()
    with open('dev.csv', 'rb') as f_dev:
        reader = csv.reader(f_dev)
        for row in reader:
            movieID = int(row[0])
            userID = int(row[1])

            # print "avg rating for this movie " + str(1.0 * sum(users_by_movieID(3)) / len(users_by_movieID(3)))
            rating = 0.0
            kNN_list = []
            if similarity_method == "dot_product":
                kNN_list = getKNN(userID, np.squeeze(np.asarray(user_similarity_dot_product[userID])), k, rating_method)

            elif similarity_method == "cosine_similarity":
                kNN_list = getKNN(userID, np.squeeze(np.asarray(user_similarity_cosine_similarity[userID])), k,
                                  rating_method)

            if kNN_list == []:
                rating = movie_avg[movieID]
            else:
                for i in range(len(kNN_list)):
                    userID_weight = kNN_list[i]
                    if (movieID, userID_weight[0]) in ratingDict:
                        # print "Within here"
                        rating += userID_weight[1] * ratingDict[(movieID, userID_weight[0])]
                    else:
                        rating += userID_weight[1] * movie_avg[movieID]
            rating += 3
            f.write(str(rating) + "\n")
            if rating < 1 or rating > 5:
                print "Warning: movie ID userID " + str(movieID) + " " + str(userID) + " " + str(rating)
    f.close()

    f = open("eval/Experiment3_" + rating_method + "_" + similarity_method + "_" + str(k)+"_predictions.txt", 'w')

    currentTime = time.time()
    with open('test.csv', 'rb') as f_dev:
        reader = csv.reader(f_dev)
        for row in reader:
            movieID = int(row[0])
            userID = int(row[1])

            # print "avg rating for this movie " + str(1.0 * sum(users_by_movieID(3)) / len(users_by_movieID(3)))
            rating = 0.0
            kNN_list = []
            if similarity_method == "dot_product":
                kNN_list = getKNN(userID, np.squeeze(np.asarray(user_similarity_dot_product[userID])), k, rating_method)

            elif similarity_method == "cosine_similarity":
                kNN_list = getKNN(userID, np.squeeze(np.asarray(user_similarity_cosine_similarity[userID])), k,
                                  rating_method)

            if kNN_list == []:
                rating = movie_avg[movieID]
            else:
                for i in range(len(kNN_list)):
                    userID_weight = kNN_list[i]
                    if (movieID, userID_weight[0]) in ratingDict:
                        # print "Within here"
                        rating += userID_weight[1] * ratingDict[(movieID, userID_weight[0])]
                    else:
                        rating += userID_weight[1] * movie_avg[movieID]
            rating += 3
            f.write(str(rating) + "\n")
            if rating < 1 or rating > 5:
                print "Warning: movie ID userID " + str(movieID) + " " + str(userID) + " " + str(rating)
    f.close()

    print "Finished this round " + str(time.time() - currentTime)
