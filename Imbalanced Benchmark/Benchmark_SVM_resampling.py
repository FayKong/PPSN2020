RANDOM_SEED_ALL = list(range(30))

names = ['ADASYN', 'SMOTE', 'NCL', 'OSS', 'SMOTEENN', 'SMOTETomek']
methods = [ADASYN(), SMOTE(),
           NeighbourhoodCleaningRule(n_neighbors=20),
           OneSidedSelection(n_neighbors=1, n_seeds_S=100),
           SMOTEENN(), SMOTETomek()]

for name, sampler in zip(names, methods):

    #name = names[0]
    #sampler = methods[0]

    performance = np.zeros(shape=(30, 5))

    for random in RANDOM_SEED_ALL:

        k = 5
        cv = StratifiedKFold(n_splits=k)

        tprs = []
        aucs = []
        precision = []
        recall = []
        f1 = []
        gmean = []
        mean_fpr = np.linspace(0, 1, 100)

        for train, test in cv.split(X, y):
            Xr, yr = sampler.fit_sample(X[train], y[train])
            #clf = DecisionTreeClassifier(min_samples_leaf=2, random_state=random).fit(Xr, yr)
            clf = svm.SVC(gamma='scale', probability=True, random_state=random)

            probas_ = clf.fit(Xr, yr).predict_proba(X[test])

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            y_test_pred = clf.predict(X[test])
            pre = metrics.precision_score(y[test], y_test_pred)  # precision
            rec = metrics.recall_score(y[test], y_test_pred)  # recall
            f11 = metrics.f1_score(y[test], y_test_pred)  # f1_score
            gm = geometric_mean_score(y[test], y_test_pred, average='binary')

            precision.append(pre)
            recall.append(rec)
            f1.append(f11)
            gmean.append(gm)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        performance[random, :] = [mean_auc, np.mean(precision), np.mean(recall), np.mean(f1), np.mean(gmean)]

    performance = pd.DataFrame(performance, columns=['AUC', 'precision', 'recall', 'F-measure', 'Gmean'])
