from classification.preprocess import *
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.linear_model import SGDClassifier

X_data, Y = load("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/nakayama.csv")
X = standartize(X_data)
X_train, X_test, y_train, y_test = train_test(X, Y)

alphas = np.logspace(-1, 0, 20)
#cs = 1/alphas
#### plot paths for elastic net and lasso
start = datetime.now()
clf = SGDClassifier(loss="log", penalty="elasticnet", alpha=0.01)#LogisticRegression(C=1.0, penalty='l1', tol=1e-6)#, multi_class="multinomial", solver="saga")
coefs_ = []
for c in alphas:
    clf.set_params(alpha=c)
    clf.fit(X_train, y_train)

    coef_to_append = clf.coef_.copy()
    coef_to_append = np.sqrt(np.sum(coef_to_append*coef_to_append, 0))
    # print(type(clf.coef_), clf.coef_.shape)
    # print(type(coef_to_append), coef_to_append.shape)

    coefs_.append(coef_to_append)
    # coefs_.append(clf.coef_.ravel().copy())
print("This took ", datetime.now() - start)


# list to array
coefs_ = np.array(coefs_)
print("coefs_.shape = ", coefs_.shape)
# find zeros
I_zeros = np.alltrue(coefs_ < 1e-6, 0)
print("number of zero features: ", sum(I_zeros))
# remove zeros
coefs_ = coefs_[:, ~I_zeros]
print("reduced coefs_.shape = ", coefs_.shape)
# further reduction
#coefs_ = coefs_[:, np.arange(0, coefs_.shape[1], 5)]

plt.plot(np.log10(alphas), coefs_)
ymin, ymax = plt.ylim()
plt.xlabel('log(lambda)')
plt.ylabel('Norm of Coefficients')
plt.title('Elastic Net Path')
plt.axis('tight')
plt.show()