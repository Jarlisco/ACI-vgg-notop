import os, sys
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score



################################
# LOAD DATASET
################################


base_dir   = "/home/odul/Documents/ESIR2/ACI/Intel/data/intel-image-classification"
output_dir="./"

X_train = np.load(output_dir + "/vgg16_features/vgg16_train_descriptors.npy")
y_train = np.load(output_dir + "/vgg16_features/vgg16_train_target.npy")

X_test = np.load(output_dir + "/vgg16_features/vgg16_test_descriptors.npy")
y_test = np.load(output_dir + "/vgg16_features/vgg16_test_target.npy")

target_names = ['buildings','forest','glacier','mountain','sea','street']


print("Class distribution in train set: \n",np.unique(y_train, return_counts=True))
print("Class distribution in test set: \n", np.unique(y_test, return_counts=True))


