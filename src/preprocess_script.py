import os
import glob
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from preprocess import parse_laser_condition, parse_name, simplify_dir_name
from preprocess import get_wanted_frames_for_condition, preprocess
from preprocess import get_average_blue_img, get_highest_power_for_cond
from preprocess import get_dir_name_from_cond
from temp_calibration import moments

### Global
x_r = (400, 800)
y_r = (400, 1000)

# Mac
#path = '/Users/mingchiang/Desktop/github/thermal-reflectance-calibration/data//0608/'
# Linux
path = '/home/mingchiang/Desktop/Data/0622/'

blank_path = '/home/mingchiang/Desktop/Data/Calibration_0622'

yaml_path = f'../data/yaml/0622.yaml'

with open(yaml_path, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
print(yaml_dict)

live_img_conds = list(yaml_dict.keys())

for idx, p in enumerate(live_img_conds):
    print(idx, p)

all_conds = [parse_laser_condition(d)
                   for d in live_img_conds]

for cond in live_img_conds[11:]: 
    dir_name = get_dir_name_from_cond(cond)
    live_img_dir = path + dir_name
    print(live_img_dir)
    wanted_frame = get_wanted_frames_for_condition(cond, yaml_dict)
    cond_dict = parse_laser_condition(dir_name)
    print(f'Working on {cond}')
    max_pw = int(get_highest_power_for_cond(cond_dict, all_conds))
  
    dw = cond_dict['dwell']
    
    blank_im = np.load(blank_path + f'/{dw}us.npy')

    if wanted_frame:
        if max_pw != cond_dict['power']:
            dw = cond_dict['dwell']
            print(f'The center is estimated by {dw}us_{max_pw}W data')
            ref_xs = np.load(f'../data/npy/{dw}us_{max_pw}W_xs.npy')
            ref_ys = np.load(f'../data/npy/{dw}us_{max_pw}W_ys.npy')
            estimate = (np.mean(ref_xs), np.mean(ref_ys))
        else:
            estimate = False
        live_im, xs, ys, pfits = preprocess(live_img_dir, wanted_frame, 
                             blank_im, x_r=x_r, y_r=y_r, blank_bypass=True, 
                             center_estimate=estimate, t=cond_dict['power'],
                             dwell=cond_dict['dwell'], plot=False, savefig=False)

        np.save(f'../data/npy/{cond}_img.npy', live_im)
        np.save(f'../data/npy/{cond}_xs.npy', xs)
        np.save(f'../data/npy/{cond}_ys.npy', ys)
        np.save(f'../data/npy/{cond}_pfit.npy', pfits)

