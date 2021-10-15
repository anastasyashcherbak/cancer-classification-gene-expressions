from classification.preprocess import *
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier


data = pd.read_csv("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/sun.csv")
data = data.iloc[:, 1:]
del data['x.43073']
##### data adjust
X_data = data.drop("y", axis=1)
Y = data["y"]
dic = {'non-tumor':0, 'astrocytomas':1, 'glioblastomas':2, 'oligodendrogliomas':3}
Y.replace(dic, inplace=True)
X = standartize(X_data)
X_train, X_test, y_train, y_test = train_test(X, Y)

clf = SGDClassifier(loss="log", penalty="elasticnet")
#clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6, multi_class="multinomial", solver="saga")
clf.fit(X_train, y_train)
coefs_ = []
coef_to_append = clf.coef_.copy()
#coefs_ = np.sqrt(np.sum(coef_to_append*coef_to_append, 0))
coefs_ = np.array(coef_to_append)
print(coefs_)
print("coefs_.shape = ", coefs_.shape)
# find zeros
I_zeros = np.alltrue(coefs_ < 1e-6, 0)
print("number of zero features: ", sum(I_zeros))
# remove zeros
coefs_ = coefs_[:, ~I_zeros]
print("reduced coefs_.shape = ", coefs_.shape)

# X_new, X_new_1, X_new_2 = feature_select(X, Y)
# #### classification
# clf = LinearSVC(C=1, multi_class='crammer_singer')
# clf1 = LinearSVC(C=1)
# clf2 = SGDClassifier(loss="log", penalty="elasticnet", alpha=0.01)
# clf3 = LogisticRegression(penalty="l2", solver='sag', max_iter=1000, random_state=42, multi_class='multinomial')
# clf6 = LogisticRegression(penalty="l1", solver='saga', max_iter=1000, random_state=42, C=10, multi_class='multinomial')
# scores = cross_val_score(clf, X, Y, cv=10)
# print(scores)
# print("Accuracy SVM CS: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# scores1 = cross_val_score(clf1, X, Y, cv=10)
# print(scores1)
# print("Accuracy SVM ovr: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
# ######
# scores2 = cross_val_score(clf2, X, Y, cv=10)
# print(scores2)
# print("Accuracy Elastic: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
# ######
# scores3 = cross_val_score(clf3, X, Y, cv=10)
# print(scores3)
# print("Accuracy LR l2 multi: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))
# ######
# scores6 = cross_val_score(clf6, X, Y, cv=10)
# print(scores6)
# print("Accuracy LR l1 mult: %0.2f (+/- %0.2f)" % (scores6.mean(), scores6.std() * 2))
#
# #### with feature selection
# scores7 = cross_val_score(clf3, X_new, Y, cv=10)
# print(scores7)
# print("Accuracy LRl2 Fscore: %0.2f (+/- %0.2f)" % (scores7.mean(), scores7.std() * 2))
# scores8 = cross_val_score(clf3, X_new_1, Y, cv=10)
# print(scores8)
# print("Accuracy LRl2 MutInfo: %0.2f (+/- %0.2f)" % (scores8.mean(), scores8.std() * 2))
# scores9 = cross_val_score(clf3, X_new_2, Y, cv=10)
# print(scores9)
# print("Accuracy LRl2 InfoGain: %0.2f (+/- %0.2f)" % (scores9.mean(), scores9.std() * 2))
# scores10 = cross_val_score(clf, X_new, Y, cv=10)
# print(scores10)
# print("Accuracy SVM CS Fscore: %0.2f (+/- %0.2f)" % (scores10.mean(), scores10.std() * 2))
# scores11 = cross_val_score(clf, X_new_1, Y, cv=10)
# print(scores11)
# print("Accuracy SVM CS MutInfo: %0.2f (+/- %0.2f)" % (scores11.mean(), scores11.std() * 2))
# scores12 = cross_val_score(clf, X_new_2, Y, cv=10)
# print(scores12)
# print("Accuracy SVM CS InfoGain: %0.2f (+/- %0.2f)" % (scores12.mean(), scores12.std() * 2))

