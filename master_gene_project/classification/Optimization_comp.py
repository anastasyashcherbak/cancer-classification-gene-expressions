from classification.preprocess import *
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm
# load preprocess data
X_data, Y = load("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/nakayama.csv")
X = standartize(X_data)

models = []
models.append(('SVM-dual-OVR(CD)', svm.LinearSVC()))
models.append(('SVM-dual-CS(CD)', svm.LinearSVC(multi_class='crammer_singer')))
models.append(('SVM(SGD)', SGDClassifier()))
#models.append(('LR(SAG)', LogisticRegression(penalty="l2", solver='sag', max_iter=1000, random_state=42, multi_class='multinomial')))
#models.append(('LR(NewtonCG)', LogisticRegression(penalty="l2", solver='newton-cg', max_iter=1000, random_state=42, multi_class='multinomial')))
#models.append(('LR(dual-CDt)', LogisticRegression(penalty="l2", dual=True, solver='liblinear', max_iter=1000, random_state=42)))
#models.append(('LR(SGD)', SGDClassifier(loss="log", penalty="l2", max_iter=1000, random_state=42)))

heldout = [0.95, 0.90, 0.75, 0.50, 0.25, 0.1]
rounds = 1
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
xx = 1. - np.array(heldout)
ttime = []
for name, model in models:
    t = time.time()
    yy = []

    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test(X, Y, size=i)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    #print(yy)
    plt.figure(1)
    plt.plot(xx, yy, label=name)
    t_final = time.time()-t
    ttime.append(t_final)
print(ttime)
    #plt.figure(2)
    #plt.plot(xx,ttime,label=name)
# test error comparison on different optimization algorithms
plt.figure(1)
plt.legend(loc="upper right")
plt.xlabel("Part of the training data")
plt.ylabel("Test Error Rate")
plt.show()

#####
fig = plt.figure(2)
ax = fig.add_subplot(111)
y_pos = np.arange(len(models))
plt.bar(y_pos , ttime, align='center')
plt.xticks(y_pos, dict(models).keys(),rotation=25)
plt.ylabel("Running time")
plt.show()