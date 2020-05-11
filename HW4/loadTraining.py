import csv
from scipy.sparse import csr_matrix
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import time
import pickle

ratingDict = dict()
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        ratingDict[(int(row[0]), int(row[1]))] = int(row[2])-3



movieIDs = set([movie_user_pair[0] for movie_user_pair in ratingDict.keys()])
print "#_movies " + str(len(movieIDs))

userIDs = set([movie_user_pair[1] for movie_user_pair in ratingDict.keys()])
print "#_users " + str(len(userIDs))


def total_rating(rating):
    return [movie_user_pair for movie_user_pair in ratingDict.keys() if ratingDict[movie_user_pair] == rating]


print "the number of times any movie was rated 1 " + str(len(total_rating(1)))
print "the number of times any movie was rated 3 " + str(len(total_rating(3)))
print "the number of times any movie was rated 5 " + str(len(total_rating(5)))
print "the number of times any movie was rated 2" + str(len(total_rating(2)))
print "the number of times any movie was rated 4" + str(len(total_rating(4)))
print "all together " + str(
    len(total_rating(1)) + len(total_rating(3)) + len(total_rating(5)) + len(total_rating(2)) + len(total_rating(4)))

print "the average movie rating across all users and movies " + str(1.0 * sum(ratingDict.values()) / len(ratingDict))


def movies_by_userID(userID):
    return [ratingDict[movie_user_pair] for movie_user_pair in ratingDict.keys() if movie_user_pair[1] == userID]


print "number of movies rated by user 4321: " + str(len(movies_by_userID(4321)))


def movies_by_userID_rating(userID, rating):
    return [movie_user_pair for movie_user_pair in ratingDict.keys() if
            movie_user_pair[1] == userID and ratingDict[movie_user_pair] == rating]


print "the number of times the user gave a 1 rating " + str(len(movies_by_userID_rating(4321, 1)))
print "the number of times the user gave a 3 rating " + str(len(movies_by_userID_rating(4321, 3)))
print "the number of times the user gave a 5 rating " + str(len(movies_by_userID_rating(4321, 5)))
print "avg movie rating for this user " + str(1.0 * sum(movies_by_userID(4321)) / len(movies_by_userID(4321)))


def users_by_movieID(movieID):
    return [ratingDict[movie_user_pair] for movie_user_pair in ratingDict.keys() if movie_user_pair[0] == movieID]


print "the number of users rating this movie " + str(len(users_by_movieID(3)))


def users_by_movieID_rating(movieID, rating):
    return [movie_user_pair for movie_user_pair in ratingDict.keys() if
            movie_user_pair[0] == movieID and ratingDict[movie_user_pair] == rating]


print "number of times the user gave a 1 rating " + str(len(users_by_movieID_rating(3, 1)))
print "number of times the user gave a 3 rating " + str(len(users_by_movieID_rating(3, 3)))
print "number of times the user gave a 5 rating " + str(len(users_by_movieID_rating(3, 5)))
print "avg rating for this movie " + str(1.0 * sum(users_by_movieID(3)) / len(users_by_movieID(3)))


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

a=M.dot(M.transpose()).todense()
print a[4321,8769],a[4321,8249]

# print M.shape[0], M.shape[1]
# print M[4321].todense().shape[0], M[4321].todense().shape[1]
#
# for i in range(M[4321].todense().shape[1]):
#     print M[4321].todense()[:,i][0,0]
# sum(M[4321])


lines = open('dev.queries', "r")

user_movie_dict = dict()

for line in lines:
    entry = line.strip(" ").split(" ")
    userID = int(entry[0])
    this_user_movie_dict = dict()
    for pair in entry[1:]:
        entry2 = pair.split(":")
        movieID = int(entry2[0])
        rating = int(entry2[1])
        this_user_movie_dict[movieID] = rating - 3  # option 2 in class
    user_movie_dict[userID] = this_user_movie_dict

print user_movie_dict[4321]

# num_user = M.shape[0]
# num_movie = M.shape[1]
# user_user_similarity_dot_product = np.zeros((num_user, num_user))
# user_user_similarity_cosine_similarity = np.zeros((num_user, num_user))
# user_norm = np.zeros(num_user)
#
# for i in range(num_user):
#     if i in user_movie_dict:
#         for value in user_movie_dict[i].values():
#             user_norm[i] += value * value
#     user_norm[i] = np.sqrt(user_norm[i])
#
# output = open('user_norm.pkl', 'wb')
# pickle.dump(user_norm, output)
# output.close()
#
# pkl_file = open('user_norm.pkl', 'rb')
# user_norm = pickle.load(pkl_file)
# pkl_file.close()
#
# for i in range(num_user):
#     print i
#     if i % 100 == 0:
#         currentTime = time.time()
#     for j in range(i):
#         if i in user_movie_dict and j in user_movie_dict:
#             for movie in user_movie_dict[i].keys():
#                 if movie in user_movie_dict[j].keys():
#                     user_user_similarity_dot_product[i, j] += user_movie_dict[i][movie] * user_movie_dict[j][movie]
#     if i % 100 == 0:
#         print "Finished 100 users " + str(time.time() - currentTime)
#     user_user_similarity_dot_product[i, i] = 0
#
# for i in range(num_user):
#     for j in range(i + 1, num_user):
#         if i in user_movie_dict and j in user_movie_dict:
#             user_user_similarity_dot_product[i, j] = user_user_similarity_dot_product[j, i]
#             user_user_similarity_cosine_similarity[i, j] = float(user_user_similarity_dot_product[i, j]) / user_norm[i] * \
#                                                            user_norm[j]
#             user_user_similarity_cosine_similarity[j, i] = user_user_similarity_cosine_similarity[i, j]
#
# output = open('user_user_similarity_dot_product.pkl', 'wb')
# pickle.dump(user_user_similarity_dot_product, output)
# output.close()
#
# pkl_file = open('user_user_similarity_dot_product.pkl', 'rb')
# user_user_similarity_dot_product = pickle.load(pkl_file)
# pkl_file.close()
#
# output = open('user_user_similarity_cosine_similarity.pkl', 'wb')
# pickle.dump(user_user_similarity_cosine_similarity, output)
# output.close()
#
# pkl_file = open('user_user_similarity_cosine_similarity.pkl', 'rb')
# user_user_similarity_cosine_similarity = pickle.load(pkl_file)
# pkl_file.close()
#
# print user_user_similarity_cosine_similarity[4321, 8249]
# print user_user_similarity_cosine_similarity[8249, 4321]


def similarity_to_userID(userID, M, topN, method):
    num_user = M.shape[0]
    similarity_dict = dict()
    for i in range(num_user):
        # print i
        similarity = 0
        if i != userID and M[i].todense().any() and M[userID].todense().any():
            if method == "dot_product":
                similarity = (M[i] * M[userID].transpose())[0, 0]
            elif method == "cosine_similarity":
                similarity = float((M[i] * M[userID].transpose())[0, 0]) / np.sqrt((M[i] * M[i].transpose())[0, 0] *
                                                                                   (M[userID] * M[userID].transpose())[
                                                                                       0, 0])

        similarity_dict[i] = similarity
        # print similarity, sum(M[i])[0, 0], sum(M[userID])[0, 0]
        # print M[userID].todense()
    return dict(sorted(similarity_dict.iteritems(), key=itemgetter(1), reverse=True)[:topN])


def similarity_to_movieID(movieID, M, topN, method):
    num_movie = M.shape[1]
    similarity_dict = dict()
    for i in range(num_movie):
        # print i
        similarity = 0.0
        if i != movieID and M[:, i].todense().any() and M[:, movieID].todense().any():
            # print "withing "
            if method == "dot_product":
                similarity = (M[:, i].transpose() * M[:, movieID])[0, 0]
            elif method == "cosine_similarity":
                similarity = float((M[:, i].transpose() * M[:, movieID])[0, 0]) / np.sqrt(
                    (M[:, i].transpose() * M[:, i])[0, 0] *
                    (M[:, movieID].transpose() * M[:, movieID])[
                        0, 0])

        similarity_dict[i] = similarity
    return dict(sorted(similarity_dict.iteritems(), key=itemgetter(1), reverse=True)[:topN])

currentTime = time.time()
print similarity_to_userID(4321, M, 5, "dot_product")
print "Finished in seconds dot_product " + str(time.time() - currentTime)

currentTime = time.time()
print similarity_to_userID(4321, M, 5, "cosine_similarity")
print "Finished in seconds cosine_similarity " + str(time.time() - currentTime)

currentTime = time.time()
print similarity_to_movieID(3, M, 5, "dot_product")
print "Finished in seconds dot_product " + str(time.time() - currentTime)

currentTime = time.time()
print similarity_to_movieID(3, M, 5, "cosine_similarity")
print "Finished in seconds cosine_similarity " + str(time.time() - currentTime)


