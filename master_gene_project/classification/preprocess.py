import itertools
import seaborn as sns
from classification.specific import *
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import PCA, KernelPCA
from scipy.cluster.hierarchy import dendrogram, linkage


##### data load
def load(file):
   # file = '../data/nakayama.csv' # data with 10 number of classes
   data = pd.read_csv(file)
   data = data.iloc[:,1:]

   data = data.drop(data.index[[35, 36, 37, 38, 39, 40, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]])
   data.index = np.arange(0, 86)
   ##### data adjust
   X_data = data.drop("y", axis=1)
   Y = data["y"]
   dic = {'synovial sarcoma':0, 'myxoid liposarcoma':1, 'dedifferentiated liposarcoma':2, 'myxofibrosarcoma':3, 'malignant fibrous histiocytoma':4}

   Y.replace(dic, inplace=True)
   return X_data, Y

#### standartization
def standartize(X_data):
   "standartize data by samples with mean 0 and variance 1"
   X = pd.DataFrame(preprocessing.scale(X_data, axis=1))
   X.columns = X_data.columns
   print(X.shape)
   return X

def train_test(X, Y, size=0.25):
   X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), Y, test_size=size, random_state=0)
   X_train = X_train.sort_index()  # inplace=True
   y_train = y_train.sort_index()
   X_test = X_test.sort_index()
   y_test = y_test.sort_index()
   return  X_train, X_test, y_train, y_test

def feature_select(X_train, y_train):
   "feature selection F-stat,mutual, infoGain"
   X_new = SelectKBest(f_classif, k=500).fit_transform(X_train, y_train)

   clf = SelectKBestCustom(mutual_info_classif, k=500)
   X_new_1 = clf.fit_transform(X_train, y_train)

   clf = ExtraTreesClassifier(criterion="entropy")
   clf = clf.fit(X_train, y_train)
   #clf.feature_importances_
   model = SelectFromModel(clf, prefit=True)
   X_new_2 = model.transform(X_train)
   print(X_new_2.shape)
   return X_new, X_new_1, X_new_2

def PCAstandard(X_train, n=15):
   sklearn_pca = sklearnPCA(n_components=n)
   Y_sklearn = sklearn_pca.fit_transform(X_train)
   Y_sklearn = pd.DataFrame(Y_sklearn)
   return Y_sklearn

def PCAkernel(X_train, kernel="rbf", n=4, gamma = 0.08, degree=3):
   kpca = KernelPCA(kernel=kernel, gamma=gamma, n_components=n, degree=degree)
   # (kernel = "poly",degree=20,n_components=30)#(kernel="rbf", gamma=0.08,n_components=20)
   X_kpca = kpca.fit_transform(X_train)
   X_kpca = pd.DataFrame(X_kpca)
   return X_kpca

def cluster(X_train, method='single'):
   Z = linkage(X_train, method)  # 'complete', 'average'
   return Z
