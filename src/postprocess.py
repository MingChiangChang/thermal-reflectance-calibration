import glob

import numpy as np
import matplotlib.pyplot as plt

def parse_output_name(output_name):
    cond = {}
    n = output_name.split('_')
    cond['dwell'] = n[0][:n[0].index('us')]
    cond['power'] = n[1][:n[1].index('W')]
    return cond

if __name__ == '__main__':
    kappa = 1.2E-4
    d = sorted(glob.glob('*_img.npy'))
    blank = np.load('../blank_0608.npy')
    for i in d:
        data = np.load(i)
        cond = parse_output_name(i)
        dwell = cond['dwell']
        power = cond['power']
        try:
            plt.imshow(data/blank/kappa)
            plt.title(f'{dwell}us {power}W')
            plt.show()
        except:
            print(f'{dwell}us {power}W seems to be an empty file')

        
