import sys
import operator

# for i in $(ls indri-lists/*); do python rerank.py "$i" "GPR-10.txt" "WS-combined.txt" ; done;
import time

currentTime = time.time()

relevance_scores = open(sys.argv[1], "r")
# print sys.argv[0], sys.argv[1],sys.argv[2]

queryID = sys.argv[1].split(".")[0]

pagerank_score_boost = 10
indri_score_boost = 0

score_dict = dict()
for line in relevance_scores:
    entry = line.strip(" ").split(" ")
    docid = int(entry[2])

    score = float(entry[4])
    score_dict[docid] = score * indri_score_boost

pagerank_scores = open(sys.argv[2], "r")

for line in pagerank_scores:

    entry = line.strip(" ").split(" ")
    # print line, sys.argv[1],sys.argv[2],sys.argv[3]
    docid = int(entry[0])
    score = float(entry[1])
    if docid in score_dict:
        score_dict[docid] += score * pagerank_score_boost

sorted_score = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)


result = open(str(indri_score_boost) + "_" + str(pagerank_score_boost) + "_" + sys.argv[3], "a")

for i in range(len(sorted_score)):
    result.write(
        queryID + " Q0 " + str(sorted_score[i][0]) + " " + str(i + 1) + " " + str(sorted_score[i][1]) + " xinq\n")

print "Finished in seconds " + str(time.time() - currentTime)
