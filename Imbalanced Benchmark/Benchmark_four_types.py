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

glass1_data = pd.read_csv('~/data/glass1.csv')
X = glass1_data.iloc[:, 0:9]


y = glass1_data['Class']
X = StandardScaler().fit_transform(X)
X = np.c_[X]


# %% HVDM metric

# HVDM metric for numerical attribute
def nor_diff(x, y):
    scale = 4 * np.nanstd(X, axis=0)
    result = np.sum(np.square(abs(x - y) / scale))
    return result

# compute the distance metric
dis_metric = np.zeros((X.shape[0], X.shape[0]))

for i in range(X.shape[0]):
    inx = [i] * X.shape[0]
    for j, p in enumerate(inx):
        dis_metric[p, j] = nor_diff(X[p], X[j])

for i, j in enumerate(range(X.shape[0])):
    dis_metric[i, j] = 1000000


num_neighbor = 5

top = np.argpartition(dis_metric, num_neighbor, axis=1)[:, :num_neighbor]
dis_neighbor = dis_metric[np.arange(dis_metric.shape[0])[:, None], top]  # compute neighbors for every sample


count_neighbor = np.zeros((X.shape[0], num_neighbor+2))  # count the types of neighbors and decide type

for i in range(count_neighbor.shape[0]):
    count_neighbor[i, :num_neighbor] = y[top[i, :]]
    count_neighbor[i, num_neighbor] = abs(5 * y[i] -
                                          np.sum(count_neighbor[i, :num_neighbor]))
    if count_neighbor[i, num_neighbor] == 5:
        count_neighbor[i, num_neighbor+1] = 0.1
    elif count_neighbor[i, num_neighbor] == 4:
        count_neighbor[i, num_neighbor+1] = 0.25
    elif count_neighbor[i, num_neighbor] == 3:
        count_neighbor[i, num_neighbor+1] = 0.4
    elif count_neighbor[i, num_neighbor] == 2:
        count_neighbor[i, num_neighbor+1] = 0.6
    elif count_neighbor[i, num_neighbor] == 1:
        count_neighbor[i, num_neighbor+1] = 0.8
    else:
        count_neighbor[i, num_neighbor+1] = 1


X_all = glass6_data
X_all['degree'] = count_neighbor[:, num_neighbor+1]
