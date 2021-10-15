from classification.preprocess import *
#from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from classification.specific import *

X_data,Y = load("C:/Users/ns/OneDrive/Documents/TUM/Master thesis/dataset_gene/master_gene_project/data/nakayama.csv")
X = standartize(X_data)

models = []
#models.append(("Fscore_SVM", Pipeline([('Fscore', SelectKBest(f_classif)), ('SVM-CS', LinearSVC(C=1, multi_class='crammer_singer'))])))
#models.append(("MutInfo_SVM", Pipeline([('MutInfo', SelectKBestCustom(mutual_info_classif)), ('SVM-CS', LinearSVC(C=1, multi_class='crammer_singer'))])))
models.append(("Fscore_LogisticRegression", Pipeline([('Fscore', SelectKBest(f_classif)), ('LogisticRegression', LogisticRegression(penalty="l2", solver='sag', max_iter=1000, random_state=42,multi_class='multinomial'))])))
models.append(("MutInfo_LogisticRegression", Pipeline([('MutInfo', SelectKBestCustom(mutual_info_classif)), ('LogisticRegression', LogisticRegression(penalty="l2", solver='sag', max_iter=1000, random_state=42,multi_class='multinomial'))])))
#selection = SelectKBest(f_classif)
#clf = Pipeline([('Fscore', selection),('SVM-CS', LinearSVC(C=1, multi_class='crammer_singer'))])

# #############################################################################
# Plot the cross-validation score as a function of percentile of features

# OLD
# score_means = list()
# score_stds = list()

score_means = {"Fscore_LogisticRegression": list(), "MutInfo_LogisticRegression": list()}
score_stds = {"Fscore_LogisticRegression": list(), "MutInfo_LogisticRegression": list()}

nfeatures = (1, 10, 50, 100, 400, 600, 900, 1000, 3000, 5000, 8000, 10000, 12000, 15000, 18000, 20000, 22283)
for name, model in models:
    for percentile in nfeatures:
        if name == "Fscore_LogisticRegression":
            model.set_params(Fscore__k=percentile)
        if name == "MutInfo_LogisticRegression":
            model.set_params(MutInfo__k=percentile)
        # Compute cross-validation score
        this_scores = cross_val_score(model, X, Y)

        # OLD
        # score_means.append(this_scores.mean())
        # score_stds.append(this_scores.std())

        score_means[name].append(this_scores.mean())
        score_stds[name].append(this_scores.std())

    # PLOT
    plt.errorbar(nfeatures, score_means[name], np.array(score_stds[name]), label=name)

plt.grid()
plt.title('Performance of the l2 regularized multinomial Logistic Regression varying the number of selected features')
plt.xlabel('Number of features')
plt.ylabel('Cross-Validation Rate')
plt.legend(loc='lower right')

plt.axis('tight')
plt.show()


