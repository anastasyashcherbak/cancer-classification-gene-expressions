import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from classification.preprocess import *
from sklearn.linear_model import SGDClassifier
# load dataset
X_data, Y = load("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/nakayama.csv")
X = standartize(X_data)
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LR_l2', LogisticRegression(penalty="l2", solver='sag', max_iter=1000, random_state=42, multi_class='multinomial')))
models.append(('LR_l2_ovr', LogisticRegression(penalty="l2", solver='sag', max_iter=1000, random_state=42)))
models.append(('LR_l1_ovr', LogisticRegression(penalty="l1", solver='saga', max_iter=1000, random_state=42, C=10)))
models.append(('Elastic_Net_mult', SGDClassifier(loss="log", penalty="elasticnet", alpha=0.01)))
models.append(('SVM_CS', LinearSVC(C=1, multi_class='crammer_singer')))
models.append(('SVM_ovr', LinearSVC(C=1)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylabel("Cross-validation accuracy")
plt.xlabel("Algorithms")
plt.show()