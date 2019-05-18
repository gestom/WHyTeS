import numpy as np

def dirrections(X, batch):
    X = appending(X, 0, batch)
    angle = np.array([np.cos(X[:, 3]), np.sin(X[:, 3])]).T
    matrix = []
    for i in xrange(1, batch + 1):
        diff = X[i:, 1:3] - X[:-i, 1:3]
        time = X[i:, 0] - X[:-i, 0]
        weight = np.sqrt(np.mean(diff**2, axis=1, keepdims=True))
        weight[(weight == 0) | (weight > 0.8)] = -0.000000000000017
        weight[(time > 1.3)] = -0.0000000000000017
        diff = diff / weight
        diff = appending(diff, i, batch)
        out = 1.2 - (np.sum(np.abs(angle - diff[:len(angle), :]), axis=1))
        out[out < 0] = 0.0
        out[out > 1] = 1.0
        matrix.append(out)
    matrix = np.array(matrix).T
    matrix = np.max(matrix, axis=1)[: 1-batch]
    return matrix


def appending(X, pos, cel):
    d = np.shape(X)
    if len(d) == 2:
        d = (1,d[1])
    else:
        d = 1
    vector = np.empty(d)
    for i in xrange(pos):
        X = np.append(vector, X, axis=0)
    for j in xrange(pos+1, cel):
        X = np.append(X, vector, axis=0)
    return X
    

batch = 20


X = np.loadtxt('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
target = np.loadtxt('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')[:, -1]
y = dirrections(X, batch)

print('CHEAT RMSE for days+nights: ' + str(np.sqrt(np.mean((y - target) ** 2.0))))
    
X = np.loadtxt('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_days_with_angles_plus_reversed.txt')
target = np.loadtxt('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_days_with_angles_plus_reversed.txt')[:, -1]
y = dirrections(X, batch)

print('CHEAT RMSE for days: ' + str(np.sqrt(np.mean((y - target) ** 2.0))))


X = np.loadtxt('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')
target = np.loadtxt('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')[:, -1]
y = dirrections(X, batch)
time = np.loadtxt('/home/tom/pro/my/WHyTeS/data/wednesday_thursday_nights_with_angles_plus_reversed.txt')[:, 0]
night = (time % 86400.0 < 3600.0*7.0) | (time % 86400.0 > 3600.0*19.0)
target = target[night]
y = y[night]

print('CHEAT RMSE for nights: ' + str(np.sqrt(np.mean((y - target) ** 2.0))))
