from classification.preprocess import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

X_data, Y = load("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/nakayama.csv")
X = standartize(X_data)
# X_new, X_new_1, X_new_2 = feature_select(X, Y)
X_train, X_test, y_train, y_test = train_test(X, Y)

#X_train, X_test, y_train, y_test, X_new, X_new_1, X_new_2 = np.load("../data/processed_data.npy")

#X_train = PCAkernel(X_train, "rbf", gamma=0.008, n=22)
#X_test = PCAkernel(X_test, "rbf", gamma=0.008, n=22)
#X_PCA = PCAstandard(X_new, n=22)
#clf = SGDClassifier(loss="log", penalty="elasticnet", alpha=0.01).fit(X_train, y_train)
#clf = LogisticRegression(penalty="l1", solver='saga', max_iter=1000, random_state=42, C=10,
#                            multi_class='multinomial').fit(X_train, y_train)
clf = LinearSVC(C=1, multi_class='crammer_singer').fit(X_train, y_train)
# y_pred = clf.predict(X_test)

#X_test_PCA = PCAstandard(X_test, n=22)
# print the training scores
#print("training score : %.3f " % clf.score(X_train, y_train))
#print("test score : %.3f " % clf.score(X_test, y_test))

y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)
class_names = np.array(['synovial s.', 'myxoid lipos.', 'dediff. lipos.', 'myxofibros.', 'malignant fibrous.'])
print(clf.coef_)


# Compute confusion matrix
cnf_matrix_test = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
cnf_matrix_train = confusion_matrix(y_train, y_pred_train)
tick_marks = np.arange(len(class_names))
# confusion matrix test
fig, ax = plt.subplots()
ax = sns.heatmap(cnf_matrix_test, annot = True, fmt = '', vmin = 0.0)
plt.title("Confusion matrix for test data")
plt.xlabel('predicted labels')
plt.ylabel('true (test) labels')
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names, rotation = 0)
plt.show()
# confusion matrix train
fig, ax = plt.subplots()
ax = sns.heatmap(cnf_matrix_train, annot = True, fmt = '', vmin = 0.0)
plt.title("Confusion matrix for train data")
plt.xlabel('predicted labels')
plt.ylabel('true (train) labels')
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names, rotation=0)
plt.show()

