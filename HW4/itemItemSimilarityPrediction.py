import csv
from scipy.sparse import csr_matrix
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import time
import pickle
# from loadTraining import users_by_movieID

from sklearn.metrics.pairwise import cosine_similarity

ratingDict = dict()
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        ratingDict[(int(row[0]), int(row[1]))] = int(row[2]) - 3  # code start imputation


def movies_by_userID(userID):
    return [ratingDict[movie_user_pair] for movie_user_pair in ratingDict.keys() if movie_user_pair[1] == userID]


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

user_avg = np.asarray(np.sum(M.todense(), axis=1, dtype=float) / M.todense().shape[0])[:,0]

currentTime = time.time()
item_similarity_dot_product = M.transpose().dot(M).todense()  # matrix A
item_similarity_dot_product = item_similarity_dot_product - (
                                                                1 - item_similarity_dot_product.min()) * np.diag(
    np.diag(item_similarity_dot_product))  # subtract self-similarity as least
item_similarity_cosine_similarity = cosine_similarity(M.transpose())

item_similarity_cosine_similarity = item_similarity_cosine_similarity - (
                                                                            1 - item_similarity_cosine_similarity.min()) * np.diag(
    np.diag(item_similarity_cosine_similarity))


def getKNN(movieID, row, k, method):
    movie_weight_list = sorted(rowToDict(row).iteritems(), key=itemgetter(1), reverse=True)[:k]

    if method == "mean":
        kNN_list = [(movie_weight[0], 1.0 / k) for movie_weight in movie_weight_list]
        return kNN_list
    elif method == "weighted_mean":
        weight_sum = sum([movie_weight[1] for movie_weight in movie_weight_list])
        if weight_sum == 0:
            return []
            print movie_weight_list
            print "Warning: " + str(weight_sum)
        kNN_list = [(movie_weight[0], movie_weight[1] / weight_sum) for movie_weight in movie_weight_list]
        return kNN_list


def rowToDict(row):
    d = dict()
    for i in range(len(row)):
        d[i] = row[i]
    return d


print "Testing"
print getKNN(3, item_similarity_cosine_similarity[3], 5, "weighted_mean")
print getKNN(3, item_similarity_cosine_similarity[3], 5, "mean")
print getKNN(3, np.squeeze(np.asarray(item_similarity_dot_product[3])), 5, "weighted_mean")
print getKNN(3, np.squeeze(np.asarray(item_similarity_dot_product[3])), 5, "mean")
print "\n"

variants = [("mean", "dot_product", 10), ("mean", "dot_product", 100), ("mean", "dot_product", 500),
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
    f = open("eval/Experiment2_" + rating_method + "_" + similarity_method + "_" + str(k), 'w')

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
                kNN_list = getKNN(movieID, np.squeeze(np.asarray(item_similarity_dot_product[movieID])), k,
                                  rating_method)

            elif similarity_method == "cosine_similarity":
                kNN_list = getKNN(movieID, item_similarity_cosine_similarity[movieID], k, rating_method)

            if kNN_list == []:
                rating = user_avg[userID]
                # 1.0 * sum(movies_by_userID(userID)) / len(movies_by_userID(userID))
            else:
                for i in range(len(kNN_list)):
                    movieID_weight = kNN_list[i]
                    if (movieID_weight[0], userID) in ratingDict:
                        "Within here"
                        rating += movieID_weight[1] * ratingDict[(movieID_weight[0], userID)]
                    else:
                        rating += movieID_weight[1] * user_avg[userID]
            rating += 3
            f.write(str(rating) + "\n")
            if rating < 1 or rating > 5:
                print "Warning: movie ID userID " + str(movieID) + " " + str(userID) + " " + str(rating)
    f.close()

    f = open("eval/Experiment2_" + rating_method + "_" + similarity_method + "_" + str(k)+"_predictions.txt", 'w')

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
                kNN_list = getKNN(movieID, np.squeeze(np.asarray(item_similarity_dot_product[movieID])), k,
                                  rating_method)

            elif similarity_method == "cosine_similarity":
                kNN_list = getKNN(movieID, item_similarity_cosine_similarity[movieID], k, rating_method)

            if kNN_list == []:
                rating = user_avg[userID]
                # 1.0 * sum(movies_by_userID(userID)) / len(movies_by_userID(userID))
            else:
                for i in range(len(kNN_list)):
                    movieID_weight = kNN_list[i]
                    if (movieID_weight[0], userID) in ratingDict:
                        "Within here"
                        rating += movieID_weight[1] * ratingDict[(movieID_weight[0], userID)]
                    else:
                        rating += movieID_weight[1] * user_avg[userID]
            rating += 3
            f.write(str(rating) + "\n")
            if rating < 1 or rating > 5:
                print "Warning: movie ID userID " + str(movieID) + " " + str(userID) + " " + str(rating)
    f.close()
    print "Finished this round " + str(time.time() - currentTime)
