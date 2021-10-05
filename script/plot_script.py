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
exec(open("insert_path.py").read())

import numpy as np
import matplotlib.pyplot as plt

from preprocess import *

p = ['Height', 'x', 'y', 'width_x', 'width_y', 'rho', 'base']
kappa = 1.2*10**-4
def plot_wafer(diameter):
    theta = np.linspace(0, 2*np.pi, 180)
    plt.plot(diameter*np.cos(theta), diameter*np.sin(theta), c='k')

yaml_path = '../data/yaml/even_temp_test.yaml'
with open(yaml_path, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

for d in yaml_dict:
    for i in range(6,7):
        try:
            pfit = np.load(f'../data/npy/{d}_pfit.npy')[:,0]
            pfit += np.load(f'../data/npy/{d}_pfit.npy')[:,6]
            pfit = pfit.tolist()
        except FileNotFoundError:
            print(f'{d}_pfit.npy not found.')
            continue

        frames = get_wanted_frames_for_condition(d, yaml_dict)

        for x, run in enumerate(frames):
            f = frames[run]
            sc = plt.scatter(np.repeat([-36+4*x], len(f)),
                        np.arange(-27, 27, 54/len(f)),
                        c=np.array(pfit[:len(f)]), s=150, vmin=300, vmax=500)
            del pfit[:len(f)]
        plot_wafer(50)
        plt.colorbar(sc)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(d)
        plt.savefig(f'heat')
        plt.show()


