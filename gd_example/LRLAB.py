import numpy as np


### GENERATE DATA FOR THE EXERCISE ###
# generate x and y
n = 750
x = np.random.rand(n, 3)
y = 0.4*np.ones(n,) + 0.1*x[:, 0] + 0.5*x[:, 1] + -0.4*x[:, 2] + 0.1*np.random.rand(n,)
y = y.reshape(n, 1)

# write data down to a file
z = np.hstack((x, y))
np.savetxt('data.txt', z, delimiter=',')

### GENERATE DATA FOR THE EXERCISE ###

### STARTING LAB SOLUTION ###

# load data and plot
data = np.loadtxt('data.txt', delimiter=',')

### Plotting code

#import matplotlib.pyplot as plt

#plt.scatter(data[:, 1], data[:, 3])
#plt.show()

### Solving regression with linear algebra
o = np.ones((n, 1))
X = np.hstack((o, data[:, 0:3]))
Y = data[:, -1]
tmp = np.dot(np.transpose(X), X)
tmp = np.dot(np.linalg.pinv(tmp), np.transpose(X))
theta = np.dot(tmp, y)
print("The parameters are:")
print(theta)

# calculate accuracy
err = Y - np.dot(X, theta)
print(np.sum(err*err)/n)

### Solving regression with gradient descent
# gradient descent
theta = np.random.rand(4, 1) - 0.5
print(theta)

def costfunc(theta):
    err = Y - np.dot(X, theta)
    return np.sum(err*err)/(2*n)

alpha = 0.00001
for i in range(1, 10000):
    err = Y - np.dot(X, theta)
    grad = X.T.dot(err)/n
    print(grad.shape)
    theta = theta - alpha * grad
    print(theta)
    print(costfunc(theta))