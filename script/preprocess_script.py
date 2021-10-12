'''
Script for preprocessing images for temperature surface fitting
Take raw reflectance and yaml file and output npy files
Each image is fitted and shifted in order to be properly stacked.
'''
import yaml
import sys

import numpy as np

sys.path.insert(0, '../src')
from preprocess import parse_laser_condition
from preprocess import get_wanted_frames_for_condition, preprocess
from preprocess import get_highest_power_for_cond
from preprocess import get_dir_name_from_cond, preprocess_by_frame

### Global
x_r = (150, 650)
y_r = (150, 1100)

# Mac
PATH = '/Users/mingchiang/Desktop/Data/black/'
BLANK_PATH = '/Users/mingchiang/Desktop/Data/Calibration_dot'
YAML_PATH = '../data/yaml/black.yaml'

# Linux
PATH = '/home/mingchiang/Desktop/Data/even_temp_test/'
BLANK_PATH = '/home/mingchiang/Desktop/Data/even_temp_test_calibration'
YAML_PATH = f'../data/yaml/even_temp_test.yaml'

with open(YAML_PATH, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
print(yaml_dict)

live_img_conds = list(yaml_dict.keys())

for idx, p in enumerate(live_img_conds):
    print(idx, p)

all_conds = [parse_laser_condition(d)
                   for d in live_img_conds]

for cond in live_img_conds: 
    dir_name = get_dir_name_from_cond(cond)
    live_img_dir = PATH + dir_name
    print(live_img_dir)
    wanted_frame = get_wanted_frames_for_condition(cond, yaml_dict)
    cond_dict = parse_laser_condition(dir_name)
    print(f'Working on {cond}')
    max_pw = int(get_highest_power_for_cond(cond_dict, all_conds))
    dw = cond_dict['dwell']
    blank_im = np.load(BLANK_PATH + f'/{dw}us.npy')

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
                             dwell=cond_dict['dwell'], plot=True, savefig=True)

        np.save(f'../data/npy/{cond}_img.npy', live_im)
        np.save(f'../data/npy/{cond}_xs.npy', xs)
        np.save(f'../data/npy/{cond}_ys.npy', ys)
        np.save(f'../data/npy/{cond}_pfit.npy', pfits)
