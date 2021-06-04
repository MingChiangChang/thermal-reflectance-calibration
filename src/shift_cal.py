import yaml
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from Block import Block
from temp_calibration import fit_with, gaussian_shift, moments, fit_center

kappa = 1.3*10**-4

#def stage_flatness_calibration()

directory = '/Users/mingchiang/Desktop/2021.03.09_calib/raw_data/'
dir_ls = glob.glob(directory + '*')

with open('yaml/stage_cal.yaml') as file:
    data_structure = yaml.load(file, Loader=yaml.FullLoader)

cond_dict = {}
for cond in data_structure:
    cs = cond.split('_')
    tpeak = int(cs[0][:-2])
    dwell = int(cs[1][:-1])
    if tpeak in cond_dict:
        cond_dict[tpeak].append(dwell)
    else:
        cond_dict[tpeak] = [dwell]
for c in cond_dict:
    print(c, cond_dict[c])

blank = plt.imread('/Users/mingchiang/Desktop/2021.03.09_calib/raw_data/5000us/55W/live_000.bmp')[:,:,2].astype(float)
for c in cond_dict:
    dw = c
    watt = cond_dict[c][-1]
    fp = f'{directory}/{dw}us/{watt}W' 
    cond = f'{dw}us_{watt}W'

    # Load Blank
    blank = plt.imread(f'{directory}/5000us/55W/live_000.bmp')[:,:,2].astype(float)

    # Load Live image
    images = []
    serial_num = []
    for fn in data_structure[cond]['set1']['live']:
        start = fn.index('_')+1
        serial_num.append(int(fn[start:start+3]))
        image = plt.imread(f'{fp}/{fn}')[:,:,2].astype(float)-blank
        image = image/blank/kappa
        #sc = plt.imshow(image)
        #plt.colorbar(sc)
        #plt.show()
        images.append(image[:700])

    # Fit center of image
    xs = []
    ys = []
    for idx, im in enumerate(images):
        x, y = fit_center(im, f'{c}us', f'{watt}W', idx)
        xs.append(x)
        ys.append(y)
   
    serial_num, xs, ys = zip(*sorted(zip(serial_num, xs, ys)))

    fig, ax1 = plt.subplots()
    ax1.scatter(serial_num, xs)
    ax1.plot(serial_num, xs, color='b')
    ax2 = plt.twinx()
    ax2.scatter(serial_num, ys, color='orange')
    ax2.plot(serial_num, ys, color='r')
    ax1.set_ylabel('X center', color='b')
    ax2.set_ylabel('Y center', color='r')
    ax2.set_xlabel('Frame')
    plt.title(f'{c}us_{watt}W')
    plt.savefig(f'{c}us_{watt}W')
    plt.close()            

