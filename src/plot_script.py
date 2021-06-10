import yaml

import numpy as np
import matplotlib.pyplot as plt

from preprocess import *

yaml_path = '../data/yaml/test.yaml'

xs = np.load('855_60_xs.npy').tolist()
ys = np.load('855_60_ys.npy').tolist()

frames = get_wanted_frames_for_condition('855us_60W', yaml_path)

for run in frames:
    f = frames[run]
    mean = np.mean(ys[:len(f)])
    new_x = [x-mean for x in ys[:len(f)]]
    plt.plot(np.arange(len(f)), new_x)
    del ys[:len(f)]
plt.xlabel('Frame #')
plt.ylabel('Shift compare to average in each scan (pxls)')
plt.show()
