import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import geometric_mean_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule, OneSidedSelection
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from scipy import interp
from os.path import join
from sklearn import svm

# %% create chess dataset

glass1_data = pd.read_csv('~data/glass1.csv')
X = glass1_data.iloc[:, 0:9]


y = glass1_data['Class']
X = StandardScaler().fit_transform(X)
X = np.c_[X]

ir = len(y[y == 1]) / len(y)
clf = LocalOutlierFactor(n_neighbors=10, contamination=ir)  # highly rely on the IR in training set
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_
X_scores_1 = np.array([X_scores[x:x + 1] for x in range(0, len(X_scores), 1)])  # change shape (n,) to (n, 1)
X_all['LOF'] = X_scores_1

#  # experiment with LOF score
X_new = X_all.drop(['Class'], axis=1)
X_add = StandardScaler().fit_transform(X_new)
X_add = np.c_[X_add]
