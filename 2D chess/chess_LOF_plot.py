# %% figure
f=open('chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'.txt','w')

for xw, yw  in zip(X,y):
	f.write(str(xw[0])+" "+str(xw[1])+" "+str(float(yw))+"\n")


n_classes = 2
plot_colors = ['#000000','#FFFFFF']
plot_step = 0.02

cmap_blw = ListedColormap(['#000000','#FFFFFF'])
cmap_blw_light = ListedColormap(['#6E6E6E','#FFFFFF'])

plt.axis("tight")

# Plot the training points
for i, color in zip([0.,1.], plot_colors):
    idx = np.where(y[train] == i)
    plt.scatter(X[train][idx, 0], X[train][idx, 1], s=20, c=color, cmap=cmap_blw, edgecolors='black')


plot_colorss = ['#569597','#a71368']
for i, color in zip([0., 1.], plot_colorss):
    idxx = np.where(y[test] == i)
    plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, c=color, cmap=cmap_blw, edgecolors=color)

error = list(np.where(y[test]!=y_test_pred))
#error[0][1]

for i in range(0, len(error[0])):
    plt.scatter(X[test][i, 0], X[test][i, 1], s=100, edgecolors='r', facecolors='none')


plt.title('Chess Classification Performance (Add = NO)')
plt.legend()
#plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()

# %% original

plt.axis("tight")

idx = np.where(y[train] == 0)
plt.scatter(X[train][idx, 0], X[train][idx, 1], s=20, c=plot_colors[0], cmap=cmap_blw, edgecolors='black')
idx1 = np.where(y[train] == 1)
plt.scatter(X[train][idx1, 0], X[train][idx1, 1], s=20, c=plot_colors[1], cmap=cmap_blw, edgecolors='black')

plot_colorss = ['#569597', '#a71368']

idxx = np.where(y[test] == 0)
plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, marker = '^', c=plot_colorss[0], cmap=cmap_blw, edgecolors=color)
idxx1 = np.where(y[test] == 1)
plt.scatter(X[test][idxx1, 0], X[test][idxx1, 1], s=20, marker = '*', c=plot_colorss[1], cmap=cmap_blw, edgecolors=color)

error = list(np.where(y[test] != y_test_pred))
# error[0][1]


type1 = plt.scatter(X[train][idx, 0], X[train][idx, 1], s=20, c=plot_colors[0], cmap=cmap_blw, edgecolors='black')
type2 = plt.scatter(X[train][idx1, 0], X[train][idx1, 1], s=20, c=plot_colors[1], cmap=cmap_blw, edgecolors='black')
type3 = plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, marker = '^', c=plot_colorss[0], cmap=cmap_blw, edgecolors=plot_colorss[0])
type4 = plt.scatter(X[test][idxx1, 0], X[test][idxx1, 1], s=20, marker = '*', c=plot_colorss[1], cmap=cmap_blw, edgecolors=plot_colorss[1])

for i in range(0, len(error[0])):
    plt.scatter(X[test][i, 0], X[test][i, 1], s=100, edgecolors='r', facecolors='none')

plt.ylim((-0.1, 1.2))
plt.title('Chess Classification Performance (Sampler = NO, Add = NO)')
plt.legend((type1, type2, type3, type4), ("majority training", "minority training", "majority test","minority test"), loc = 0, fontsize = 'medium', framealpha = 0.2)
# plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()


# %% add

plt.axis("tight")

idx = np.where(y[train] == 0)
plt.scatter(X[train][idx, 0], X[train][idx, 1], s=20, c=plot_colors[0], cmap=cmap_blw, edgecolors='black')
idx1 = np.where(y[train] == 1)
plt.scatter(X[train][idx1, 0], X[train][idx1, 1], s=20, c=plot_colors[1], cmap=cmap_blw, edgecolors='black')

plot_colorss = ['#569597', '#a71368']

idxx = np.where(y[test] == 0)
plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, marker = '^', c=plot_colorss[0], cmap=cmap_blw, edgecolors=color)
idxx1 = np.where(y[test] == 1)
plt.scatter(X[test][idxx1, 0], X[test][idxx1, 1], s=20, marker = '*', c=plot_colorss[1], cmap=cmap_blw, edgecolors=color)

error = list(np.where(y[test] != y_test_pred))
# error[0][1]


type1 = plt.scatter(X[train][idx, 0], X[train][idx, 1], s=20, c=plot_colors[0], cmap=cmap_blw, edgecolors='black')
type2 = plt.scatter(X[train][idx1, 0], X[train][idx1, 1], s=20, c=plot_colors[1], cmap=cmap_blw, edgecolors='black')
type3 = plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, marker = '^', c=plot_colorss[0], cmap=cmap_blw, edgecolors=plot_colorss[0])
type4 = plt.scatter(X[test][idxx1, 0], X[test][idxx1, 1], s=20, marker = '*', c=plot_colorss[1], cmap=cmap_blw, edgecolors=plot_colorss[1])

for i in range(0, len(error[0])):
    plt.scatter(X[test][i, 0], X[test][i, 1], s=100, edgecolors='r', facecolors='none')

plt.ylim((-0.1, 1.2))
plt.title('Chess Classification Performance (Sampler = NO, Add = YES)')
plt.legend((type1, type2, type3, type4), ("majority training", "minority training", "majority test","minority test"), loc = 0, fontsize = 'medium', framealpha = 0.2)
# plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()


# %% original SMOTE

plt.axis("tight")

idx = np.where(yr == 0)
plt.scatter(Xr[idx, 0], Xr[idx, 1], s=20, c=plot_colors[0], cmap=cmap_blw, edgecolors='black')
idx1 = np.where(yr == 1)
plt.scatter(Xr[idx1, 0], Xr[idx1, 1], s=20, c=plot_colors[1], cmap=cmap_blw, edgecolors='black')

plot_colorss = ['#569597', '#a71368']

idxx = np.where(y[test] == 0)
plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, marker = '^', c=plot_colorss[0], cmap=cmap_blw, edgecolors=color)
idxx1 = np.where(y[test] == 1)
plt.scatter(X[test][idxx1, 0], X[test][idxx1, 1], s=20, marker = '*', c=plot_colorss[1], cmap=cmap_blw, edgecolors=color)

error = list(np.where(y[test] != y_test_pred))
# error[0][1]


type1 = plt.scatter(Xr[idx, 0], Xr[idx, 1], s=20, c=plot_colors[0], cmap=cmap_blw, edgecolors='black')
type2 = plt.scatter(Xr[idx1, 0], Xr[idx1, 1], s=20, c=plot_colors[1], cmap=cmap_blw, edgecolors='black')
type3 = plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, marker = '^', c=plot_colorss[0], cmap=cmap_blw, edgecolors=plot_colorss[0])
type4 = plt.scatter(X[test][idxx1, 0], X[test][idxx1, 1], s=20, marker = '*', c=plot_colorss[1], cmap=cmap_blw, edgecolors=plot_colorss[1])

for i in range(0, len(error[0])):
    plt.scatter(X[test][i, 0], X[test][i, 1], s=100, edgecolors='r', facecolors='none')

plt.ylim((-0.1, 1.2))
plt.title('Chess Classification Performance (Sampler = SMOTE, Add = NO)')
plt.legend((type1, type2, type3, type4), ("majority training", "minority training", "majority test","minority test"), loc = 0, fontsize = 'medium', framealpha = 0.2)
# plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()


# %% add SMOTE

plt.axis("tight")

idx = np.where(yr == 0)
plt.scatter(Xr[idx, 0], Xr[idx, 1], s=20, c=plot_colors[0], cmap=cmap_blw, edgecolors='black')
idx1 = np.where(yr == 1)
plt.scatter(Xr[idx1, 0], Xr[idx1, 1], s=20, c=plot_colors[1], cmap=cmap_blw, edgecolors='black')

plot_colorss = ['#569597', '#a71368']

idxx = np.where(y[test] == 0)
plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, marker = '^', c=plot_colorss[0], cmap=cmap_blw, edgecolors=color)
idxx1 = np.where(y[test] == 1)
plt.scatter(X[test][idxx1, 0], X[test][idxx1, 1], s=20, marker = '*', c=plot_colorss[1], cmap=cmap_blw, edgecolors=color)

error = list(np.where(y[test] != y_test_pred))
# error[0][1]

type1 = plt.scatter(Xr[idx, 0], Xr[idx, 1], s=20, c=plot_colors[0], cmap=cmap_blw, edgecolors='black')
type2 = plt.scatter(Xr[idx1, 0], Xr[idx1, 1], s=20, c=plot_colors[1], cmap=cmap_blw, edgecolors='black')
type3 = plt.scatter(X[test][idxx, 0], X[test][idxx, 1], s=20, marker = '^', c=plot_colorss[0], cmap=cmap_blw, edgecolors=plot_colorss[0])
type4 = plt.scatter(X[test][idxx1, 0], X[test][idxx1, 1], s=20, marker = '*', c=plot_colorss[1], cmap=cmap_blw, edgecolors=plot_colorss[1])

for i in range(0, len(error[0])):
    plt.scatter(X[test][i, 0], X[test][i, 1], s=100, edgecolors='r', facecolors='none')

plt.ylim((-0.1, 1.2))
plt.title('Chess Classification Performance (Sampler = SMOTE, Add = YES)')
plt.legend((type1, type2, type3, type4), ("majority training", "minority training", "majority test","minority test"), loc = 0, fontsize = 'medium', framealpha = 0.2)
# plt.savefig('plots/chess'+str(div)+'x'+str(div)+'_n'+str(N)+'_'+str(per)+'DT.eps')
plt.show()
