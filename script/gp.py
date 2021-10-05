"""
Test script for pre-fitting dwell, power vs tpeak data
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import numpy as np

dw = np.load("../data/dw.npy")
pw = np.load("../data/temp.npy")[1]
tpeak = np.load("../data/temp.npy")[2]

cond = np.array([[dw[i], pw[i]] for i in range(dw.shape[0])])

kernel = RBF([0.1,1], (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
gp.fit(cond, tpeak)

#x = [[np.log10(1000), 50], [np.log10(5000), 60]]
x = np.arange(2, 4, 0.1)
y = np.arange(30, 80, 3)
tt = x.shape[0]*y.shape[0]
xx, yy = np.meshgrid(x, y)
xvec = np.ravel(xx)
yvec = np.ravel(yy)
c = np.array([[xvec[i], yvec[i]] for i in range(tt)])
y_pred, _ = gp.predict(c, return_std=True)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(dw, pw, tpeak)
ax.plot_surface(xx, yy, y_pred.reshape(xx.shape))
plt.show()
