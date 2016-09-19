from scipy.sparse import csr_matrix
import numpy as np


# Load development data and for corpus exploration
# Usage: python load_docVec.py [.docVec file]"
def load_customizedDocVec(docVecFile, dfFile):
    row = list()
    col = list()
    data = list()
    docVectorFile = open(docVecFile, "r")

    dict = {}
    dfFile = open(dfFile, "r")
    for line in dfFile:
        entry = line.strip(" ").split(":")
        dict[int(entry[0])] = int(entry[1])

    print len(dict)

    lineTotal = 0
    for line in docVectorFile:
        lineTotal += 1
    print lineTotal
    docVectorFile = open(docVecFile, "r")

    lineCount = 0
    for line in docVectorFile:
        words = line[:-1].strip(" ").split(" ")
        for word in words:
            entry = word.split(":")
            # print entry
            wordIndex = int(entry[0])
            frequency = float(1.0 * int(entry[1]) * np.log(lineTotal/dict[wordIndex]))
            row.append(lineCount)
            col.append(wordIndex)
            data.append(frequency)
        lineCount += 1

    M = csr_matrix((np.asarray(data), (np.asarray(row), np.asarray(col))))

    print "Development set statistics"

    n = M.shape[0]  # number of documents
    print "Total number of documents " + str(n)
    print "Total number of words " + str(M.sum())
    d = M.shape[1]  # number of features/words/dimensions
    
    return n, d, M
