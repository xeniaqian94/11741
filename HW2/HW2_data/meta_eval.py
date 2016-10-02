import os
import eval_module
import numpy as np
import matplotlib.pyplot as plt

n_cluster = [67]
meta_F1=np.zeros(len(n_cluster))

for k in range(len(n_cluster)):
    F1 = np.zeros(10)
    for time in range(10):
        F1[time] = eval_module.eval("HW2_dev.eval_output_" + str(n_cluster[k]) + "_" + str(time) + "_kmeans++Revisit", "HW2_dev.gold_standards")
        print F1[time]
    print str(n_cluster[k])+" clusters: "+str(np.mean(F1))+" "+str(np.var(F1))
    meta_F1[k]=np.mean(F1)

# plt.title("Baseline k-means #_clusters v.s. F1 mean")
# plt.plot(n_cluster,meta_F1,'ro-')
# plt.grid()
# plt.show()