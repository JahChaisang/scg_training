import numpy as np

# load data and plot

# load data and plot
data = np.loadtxt('data.txt', delimiter=',')
m = data.shape[0]
n = data.shape[1] - 1

### Plotting code

#import matplotlib.pyplot as plt

#plt.scatter(data[:, 1], data[:, 3])
#plt.show()

### Solving regression with linear algebra
o = np.ones((m, 1))
x = np.hstack((o, data[:, 0:3]))
y = data[:, -1]

tmp = np.dot(np.transpose(x), x)
tmp = np.dot(np.linalg.pinv(tmp), np.transpose(x))
theta = np.dot(tmp, y)
print("Linear Algebra: the parameters")
print(theta)
# calculate accuracy
err = y - np.dot(x, theta)
print("Linear Algebra: cost = " + str(np.sum(err*err)/(2*m)) + '\n\n')

def grad(x, y, theta):
    gradient = np.zeros(n+1)
    pred = np.dot(x, theta)
    error = pred - y
    for i in range(n+1):
        gradient[i] = np.sum(x[:, i] * error)/m
    return gradient

def costfunc(x, y, theta):
    pred = np.dot(x, theta)
    error = pred - y
    return np.sum(error ** 2) / (2 * m)

alpha = 0.001
iters = 5000

theta = np.random.rand(n+1)
for i in range(iters):
    gradient = grad(x, y, theta)
    theta = theta - alpha * gradient

print("Gradient Descent: the parameters")
print(theta)
print("Linear Algebra: cost = " + str(costfunc(x, y, theta)))