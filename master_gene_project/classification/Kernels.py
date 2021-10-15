from classification.preprocess import *
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

data = pd.read_csv("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/sun.csv")
#data = pd.read_csv("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/sorlie.csv")
#data = pd.read_csv("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/chin.csv")
#data = pd.read_csv("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/shipp.csv")
#data = pd.read_csv("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/yeoh.csv")
data = data.iloc[:, 1:]
del data['x.43073']
##### data adjust
X_data = data.drop("y", axis=1)
Y = data["y"]
dic = {'non-tumor':0, 'astrocytomas':1, 'glioblastomas':2, 'oligodendrogliomas':3} #glioma
#dic = {'1': 0, '2': 1,'3': 2, '4': 3, '5': 4} #sorlie
#dic = {'negative': 0, 'positive': 1} #chin
#dic = {'DLBCL': 0, 'FL': 1} #shipp
#dic = {'BCR': 0, 'E2A': 1, 'Hyperdip': 2, 'MLL': 3, 'T': 4, 'TEL': 5} #yeoh
Y.replace(dic, inplace=True)
X = standartize(X_data)
#### load data
#X_data,Y = load("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/nakayama.csv")
#X = standartize(X_data)
#X_new, X_new_1, X_new_2 = feature_select(X, Y)
############ kernel SVM
clf = SVC(C=1, kernel='rbf', gamma=0.001, decision_function_shape='ovr')
scores = cross_val_score(clf, X, Y, cv=10)
print(scores)
print("Accuracy SVM with rbf kernel: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
###### svm with kernel PCA
# X_PCA_kernel = PCAkernel(X, "rbf", gamma=0.008, n=50)
# clf1 = LinearSVC(C=1, multi_class='crammer_singer')
# scores1 = cross_val_score(clf1, X_PCA_kernel, Y, cv=10)
# print(scores1)
# print("Accuracy SVM with rbf kernel with PCA kernel: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
# ###### SVM with PCA
# X_PCA = PCAstandard(X, n=50)
# clf2 = LinearSVC(C=1, multi_class='crammer_singer')
# scores2 = cross_val_score(clf2, X_PCA, Y, cv=10)
# print(scores2)
# print("Accuracy SVM with PCA: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
# ####### Logistic with PCA
#
# clf3 = LogisticRegression(penalty="l2", solver='sag', max_iter=1000, random_state=42, multi_class='multinomial')
# scores3 = cross_val_score(clf3, X_PCA, Y, cv=10)
# print(scores3)
# print("Accuracy LR with PCA: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))
#
# ####### Logistic with PCA
#
# #clf3 = LogisticRegression(penalty="l2", solver='sag', max_iter=1000, random_state=42, multi_class='multinomial')
# scores4 = cross_val_score(clf3, X_PCA_kernel, Y, cv=10)
# print(scores4)
# print("Accuracy LR with kernel PCA: %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std() * 2))

