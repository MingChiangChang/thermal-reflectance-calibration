'''
0: Height
1: x
2: y
3: width_x
4: width_y
5: rho
6: base
'''

import yaml

import numpy as np
import matplotlib.pyplot as plt

from preprocess import *

yaml_path = '../data/yaml/0618.yaml'
with open(yaml_path, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

for d in yaml_dict:

    #xs = np.load(f'../data/npy/{d}_xs.npy').tolist()
    #ys = np.load(f'../data/npy/{d}_ys.npy').tolist()
    try:
        pfit = np.load(f'../data/npy/{d}_pfit.npy')[:,5].tolist()
    except FileNotFoundError:
        print(f'{d}_pfit.npy not found.')
        continue

    frames = get_wanted_frames_for_condition(d, yaml_dict)

    for run in frames:
        f = frames[run]
        #mean = np.mean(ys[:len(f)])
        #new_x = [x-mean for x in ys[:len(f)]]
        plt.plot(np.arange(len(f)), pfit[:len(f)])
        del pfit[:len(f)]

    plt.xlabel('Frame #')
    plt.ylabel('rho')
    plt.title(d)
    plt.savefig(f'{d}_rho')
    plt.show()
