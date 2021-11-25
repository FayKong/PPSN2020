# %% calculate four types of minority and majority samples

def nor_diff(x, y):
    scale = 4 * np.nanstd(X, axis=0)
    result = np.sum(np.square(abs(x - y) / scale))
    return result


dis_metric = np.zeros((X.shape[0], X.shape[0]))

for i in range(X.shape[0]):
    inx = [i] * X.shape[0]
    for j, p in enumerate(inx):
        dis_metric[p, j] = nor_diff(X[p], X[j])

for i, j in enumerate(range(X.shape[0])):
    dis_metric[i, j] = 1000000

num_neighbor = 5

top = np.argpartition(dis_metric, num_neighbor, axis=1)[:, :num_neighbor]
dis_neighbor = dis_metric[np.arange(dis_metric.shape[0])[:, None], top]

count_neighbor = np.zeros((X.shape[0], num_neighbor+2))

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


y_1 = np.array([y[x:x+1] for x in range(0, len(y), 1)])
X_all = np.concatenate([X, y_1], axis=1)
X_all = pd.DataFrame(X_all, columns=['input1', 'input2', 'Class'])

X_all['degree'] = count_neighbor[:, num_neighbor+1]
maj = X_all[X_all['Class'] == 0]
min = X_all[X_all['Class'] == 1]

# maj_min = pd.concat([maj, min[min['degree']!=1]])
# min_maj = pd.concat([min, maj[maj['degree']!=1]])
