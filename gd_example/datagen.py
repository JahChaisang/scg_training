import numpy as np

### GENERATE DATA FOR THE EXERCISE ###
from sklearn.datasets.samples_generator import make_regression
n = 3
m = 750
x, y = make_regression(n_samples=m,
                       n_features=n,
                       n_informative=n,
                       noise=10,
                       random_state=2015)

newy = y.reshape(m, 1)
# write data down to a file
z = np.hstack((x, newy))
np.savetxt('data.txt', z, delimiter=',')