"""
Test script for gp fitting blanks
"""
# pylint: disable=E1101
from pathlib import Path
import os
import yaml

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2

home = Path.home()
path = home / 'Desktop' / 'Data' / 'Calibration_Chess_diode' 
YAML_PATH = '../data/yaml/Calibration_Chess_diode.yaml'

with open(YAML_PATH, 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

paths = sorted(list(path.glob('*')))

for path in tqdm(paths):
    dir_name = os.path.basename(str(path))
    dwell =int(dir_name[:dir_name.index('us')])

    frame_num = data[str(dwell) + 'us'][0]

    print(dwell)
    print(list(path.glob('*')))

    blank = plt.imread(str(path) + f'/Run-0000_LED-On_Power-Off_Frame-{str(frame_num).zfill(4)}.png')[:,:,2]

    low_res_blank = cv2.resize(blank, (50, 40), interpolation = cv2.INTER_AREA)
    #plt.imshow(low_res_blank)
    #plt.show()
    kernel = RBF([20,20], (10, 1e2))
    x, y = np.indices(low_res_blank.shape)
    low_res_blank_1d = np.ravel(low_res_blank)
    x = np.ravel(x)
    y = np.ravel(y)
    indices = np.array([[x[i], y[i]] for i in range(x.shape[0])])
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
    print("start fiting")
    gp.fit(indices, low_res_blank_1d)

    #x = [[np.log10(1000), 50], [np.log10(5000), 60]]
    #x = np.arange(2, 4, 0.1)
    #y = np.arange(30, 80, 3)
    #tt = x.shape[0]*y.shape[0]
    #xx, yy = np.meshgrid(x, y)
    #xvec = np.ravel(xx)
    #yvec = np.ravel(yy)
    #c = np.array([[xvec[i], yvec[i]] for i in range(tt)])
    y_pred, _ = gp.predict(indices, return_std=True)
    #x, y = np.indices(low_res_blank.shape)
    ##
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ##
    #ax.scatter(x, y, low_res_blank)
    #ax.plot_surface(x, y, y_pred.reshape(low_res_blank.shape))
    #plt.show()
    y_pred = y_pred.reshape((40, 50))
    final = cv2.resize(y_pred, (1280, 1024))
    np.save(str(dwell), final)
