# %% experiment with LOF score

clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(X)
n_errors = (y_pred != y).sum()
X_scores = clf.negative_outlier_factor_
Score_qu = np.quantile(X_scores, 0.12)  # find the quantile of outlier score (the least 12%)

n_classes = 2
plot_colors = ['#000000', '#FFFFFF']
plot_step = 0.02

cmap_blw = ListedColormap(['#000000', '#FFFFFF'])
cmap_blw_light = ListedColormap(['#6E6E6E', '#FFFFFF'])

# plot with all the outlier score (showing as radius)

plt.axis("tight")
# Plot the training points
for i, color in zip([0.,1.], plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], s=20, c=color, cmap=cmap_blw, edgecolors='black')

radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
#plt.scatter(X[i, 0], X[i, 1], s=1000 * radius, edgecolors='r', facecolors='none')
plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r', facecolors='none')

plt.title('Chess'+str(div)+'x'+str(div)+' Imbalanced data-set')
plt.legend()
#plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()

# %% plot with leass-than-quantile outlier score (showing as radius)

X_scores_1 = np.array([X_scores[x:x+1] for x in range(0, len(X_scores), 1)])
X_all['LOF'] = X_scores_1
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
radius_1 = np.array([radius[x:x+1] for x in range(0, len(radius), 1)])
X_all['radius'] = radius_1
LOF_sort = X_all.sort_values(by=['LOF'])
maj_safe_idx = LOF_sort[(LOF_sort['degree'] == 1) & (LOF_sort['Class'] == 0)].index
X_drop_safe_maj = LOF_sort.drop(index=maj_safe_idx)


plt.axis("tight")
# Plot the training points
for i, color in zip([0.,1.], plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], s=20, c=color, cmap=cmap_blw, edgecolors='black')

# for i in range(0, LOF_sort.shape[0]):
    # if X_scores[i] < Score_qu:
        # if X_all.iloc[i, 3] != 1:
                # plt.scatter(X[i, 0], X[i, 1], s=1000 * radius[i], edgecolors='r', facecolors='none')

t = 0
for i in range(0, X_drop_safe_maj.shape[0]):
    if t < 0.12*LOF_sort.shape[0]:  # # 0.12 should be defined
        plt.scatter(X_drop_safe_maj.iloc[i, 0], X_drop_safe_maj.iloc[i, 1],
                    s=1000 * X_drop_safe_maj.iloc[i, 5], edgecolors='r', facecolors='none')
        t = t + 1

plt.title('Chess'+str(div)+'x'+str(div)+' Class-imbalance Classification with edited LOF score')
plt.legend()
#plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()
