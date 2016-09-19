Thanks for reading this. For any questions, please feel free to contact xinq@cs.cmu.edu

This README provides instructions on how to run experiments in 6.1.2. First enter this directory. 

0. [!IMPORTANT!] It is very UNLIKELY to have scikit-learn installed on the remote UNIX environment. Please install as follows,
cd scikit-learn
make 

#This may take several minutes to finish. Build is only successful when you see the following message:

-bash-4.2$ cd ../
-bash-4.2$ python
Python 2.7.5 (default, Oct 11 2015, 17:47:16) 
[GCC 4.8.3 20140911 (Red Hat 4.8.3-9)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> from sklearn.preprocessing import normalize


1. Baseline approach, 
command: python main_baseline.py HW2_dev.docVectors
10 outputs are named as, HW2_dev.eval_output_67_[0-9]_baseline

2. k-means++ approach,
command: python main_kmeans++.py HW2_dev.docVectors
10 outputs are named as, HW2_dev.eval_output_67_[0-9]_kmeans++

3. Custom algorithm (best performing one):
on development set: 
	command: python main_Custom.py HW2_dev.docVectors HW2_dev.df
10 outputs are named as, HW2_dev.eval_output_67_[0-9]_custom_logScale
on test set: 
	command: python main_Custom.py HW2_test.docVectors HW2_test.df
10 outputs are named as, HW2_test.eval_output_67_[0-9]_custom_logScale

4. Evaluation follows the eval.py command, where you specify the clustering output file name, like,
python eval.py HW2_dev.eval_output_67_3_custom_logScale HW2_dev.gold_standards
