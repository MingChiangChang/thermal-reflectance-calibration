import os
import glob
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from preprocess import parse_laser_condition, parse_name, simplify_dir_name
from preprocess import get_wanted_frames_for_condition, preprocess
from preprocess import get_average_blue_img, get_highest_power_for_cond
from temp_calibration import moments

### Global
x_r = (400, 800)
y_r = (400, 1000)

# Mac
#path = '/Users/mingchiang/Desktop/github/thermal-reflectance-calibration/data//0608/'
# Linux
path = '/home/mingchiang/Desktop/Data/0609/'


def on_0608(dn):
    ''' Special function because our scans are done on different day'''
    cond = parse_laser_condition(dn)
    return cond['dwell']<=1941 or (cond['dwell']==2924 and cond['power']<=35)

live_img_dirs = sorted(glob.glob(path+'*'))
#live_img_dirs = [f'{path}00855us_060.00W']


# TODO move this to somewhere else as a function
#dark_imgs_0609_paths = glob.glob(f'{path}Calibration_0609/10000us_000.00W/*')
#dark_imgs_0609 = []
#for dark_img in dark_imgs_0609_paths:
#    cond = parse_name(os.path.basename(dark_img))
#    if cond['LED'] and not cond['Laser']:
#        dark_imgs_0609.append(dark_img)

#dark_imgs_0609 = #glob.glob()
blank_img_0608 = np.load('blank_0608.npy')#get_average_blue_img(dark_imgs_0608)
#dark_img_0609 = get_average_blue_img(dark_imgs_0609)
#plt.imshow(dark_img_0609)
#plt.show()
#np.save('blank_0609.npy', dark_img_0609)
blank_img_0609 = np.load('blank_0609.npy')

yaml_path = f'../data/yaml/0609.yaml'

with open(yaml_path, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

for idx, p in enumerate(live_img_dirs[::-1]):
    print(idx, p)

all_conds = [parse_laser_condition(os.path.basename(d))
                   for d in live_img_dirs]
for live_img_dir in live_img_dirs[::-1][11:12]: 
    dir_name = os.path.basename(live_img_dir)
    cond = simplify_dir_name(dir_name)
    wanted_frame = get_wanted_frames_for_condition(cond, yaml_dict)
    cond_dict = parse_laser_condition(dir_name)
    print(f'Working on {cond}')
    max_pw = int(get_highest_power_for_cond(cond_dict, all_conds))
    if wanted_frame:
        if max_pw != cond_dict['power']:
            dw = cond_dict['dwell']
            print(f'The center is estimated by {dw}us_{max_pw}W data')
            ref_xs = np.load(f'../data/npy/{dw}us_{max_pw}W_xs.npy')
            ref_ys = np.load(f'../data/npy/{dw}us_{max_pw}W_ys.npy')
            estimate = (np.mean(ref_xs), np.mean(ref_ys))
        else:
            estimate = False
        if on_0608(os.path.basename(live_img_dir)):
            live_im, xs, ys = preprocess(live_img_dir, wanted_frame, 
                                 blank_img_0608, x_r=x_r, y_r=y_r, blank_bypass=True, 
                                 center_estimate=estimate, t=cond_dict['power'],
                                 dwell=cond_dict['dwell'], plot=True, savefig=True)
        else:
            live_im, xs, ys = preprocess(live_img_dir, wanted_frame,
                                 blank_img_0609, x_r=x_r, y_r=y_r, blank_bypass=True,
                                 center_estimate=estimate, t=cond_dict['power'],
                                 dwell=cond_dict['dwell'], plot=True, savefig=True)
    #plt.imshow(live_im)
    #plt.show()

        np.save(f'../data/npy/{cond}_img.npy', live_im)
        np.save(f'../data/npy/{cond}_xs.npy', xs)
        np.save(f'../data/npy/{cond}_ys.npy', ys)

# Storing
