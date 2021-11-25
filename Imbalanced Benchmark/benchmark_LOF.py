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
