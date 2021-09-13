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

p = ['Height', 'x', 'y', 'width_x', 'width_y', 'rho', 'base']

yaml_path = '../data/yaml/even_temp_test.yaml'
with open(yaml_path, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

for d in yaml_dict:

    #xs = np.load(f'../data/npy/{d}_xs.npy').tolist()
    #ys = np.load(f'../data/npy/{d}_ys.npy').tolist()
    for i in range(5,6):
        try:
            pfit = np.load(f'../data/npy/{d}_pfit.npy')[:,i].tolist()
        except FileNotFoundError:
            print(f'{d}_pfit.npy not found.')
            continue

        frames = get_wanted_frames_for_condition(d, yaml_dict)

        for x, run in enumerate(frames):
            f = frames[run]
            #mean = np.mean(ys[:len(f)])
            #new_x = [x-mean for x in ys[:len(f)]]
            #plt.plot(np.arange(len(f)), pfit[:len(f)])
            print(len(f))
            print(np.repeat([-36+4*x], len(f)).shape,
                  np.arange(-27, 27, 54/len(f)).shape) 
            sc = plt.scatter(np.repeat([-36+4*x], len(f)),
                        np.arange(-27, 27, 54/len(f)),
                        c=pfit[:len(f)], s=150)#, vmin=0.028, vmax=0.04)
            del pfit[:len(f)]

        plt.colorbar(sc)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(d)
        plt.savefig(f'heat')
        plt.show()


if __name__ == '__main__':
    main()
