from scipy.sparse import csr_matrix
import numpy as np
import sys
import readTransitionMatrix
import scipy.stats as ss
import powerIteration
import topicSpecificTeleportationVector

import time

currentTime = time.time()
print sys.argv[1]
alpha = 0.8
beta = 0.2
gamma = 0.0
iter = 500

M = readTransitionMatrix.readTransitionMatrix(sys.argv[1])

p_0 = np.ones(M.shape[0])
p_0 = p_0 / M.shape[0]

p_t_alltopics = topicSpecificTeleportationVector.topicSpecificTeleportationVector("doc_topics.txt", M.shape[0])
num_Topics = p_t_alltopics.shape[1]

r_t_alltopics = np.zeros((M.shape[0], num_Topics))

for topic in range(num_Topics):
    r = np.zeros(M.shape[0])
    r[0] = 1

    for i in range(iter):
        print "Within " + str(i + 1) + " iteration! for topic " + str(topic + 1)
        rank_old = ss.rankdata(r, method='ordinal')

        r_new = powerIteration.powerIteration(alpha, beta, gamma, M, r, p_0, p_t_alltopics[..., topic])
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

    r_t_alltopics[..., topic] = r

query_topic_distros = open("user-topic-distro.txt", "r")

for query_topic_distro in query_topic_distros:
    r_TSPR = np.zeros(M.shape[0])
    entry = query_topic_distro.split(" ")
    user_id = int(entry[0])
    query_id = int(entry[1])
    for topic_distro in entry[2:]:
        topic_id = int(topic_distro.split(":")[0])
        prob = float(topic_distro.split(":")[1])
        r_TSPR = r_TSPR + prob * r_t_alltopics[..., (topic_id - 1)]
    writeFile = file("user-specific-score/" + str(user_id) + "-" + str(query_id) + ".results.txt", "w")
    for i in range(len(r_TSPR)):
        writeFile.write(str(i + 1) + " " + str(r_TSPR[i]) + "\n")
    writeFile.close()

print "Finished in seconds " + str(time.time() - currentTime)
