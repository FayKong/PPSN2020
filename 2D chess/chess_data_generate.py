# %% generate 4*4 chess dataset

RANDOM_SEED = 123123   # fix the seed on each iteration
np.random.seed(RANDOM_SEED)

# Parameters
div = 4   # int(sys.argv[1])
N = 1000   # int(sys.argv[2])
per = 0.1   # float(sys.argv[3])

lim = []
for i in range(1, div):
	lim.append(i/float(div))
print(lim)

X = []
Y = []
Xo = []
Yo = []
# Generation of the Synthetic data
for i in range(0, N):
	x = [0, 0]
	x[0] = np.random.uniform(0, 1)
	x[1] = np.random.uniform(0, 1)

	y = 0

	ind = 0
	while ind < len(lim) and x[0] > lim[ind]:
		ind += 1

	y = (ind)%2

	ind = 0
	while ind < len(lim) and x[1] > lim[ind]:
		ind += 1

	if y == 0:
		y = (ind)%2
	else:
		y = (ind-1)%2

	Xo.append(x)
	Yo.append(y)
	if y == 0 or (y == 1 and np.random.uniform(0, 1) < per):
		X.append(x)
		Y.append(y)


X = np.array(X)
y = np.array(Y)
Xo = np.array(Xo)
yo = np.array(Yo)

n_classes = 2
plot_colors = ['#000000', '#FFFFFF']
plot_step = 0.02

cmap_blw = ListedColormap(['#000000', '#FFFFFF'])
cmap_blw_light = ListedColormap(['#6E6E6E', '#FFFFFF'])

x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
