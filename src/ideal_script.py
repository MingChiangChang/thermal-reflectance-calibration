import os
import glob
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from preprocess import * # Temperary

path = '/Users/mingchiang/Desktop/github/thermal-reflectance-calibration/data/'

def on_0608(dn):
    ''' Special function because our scans are done on different day'''
    cond = parse_laser_condition(dn)
    return cond['dwell']<=1941 or (cond['dwell']==2924 and cond['power']<=35)

#live_img_dirs = glob.glob()
live_img_dirs = [f'{path}00855us_060.00W']

dark_imgs_0608_paths = glob.glob(f'{path}Calibration_0608/10000us_000.00W/*')
dark_imgs_0608 = []
for dark_img in dark_imgs_0608_paths:
    cond = parse_name(os.path.basename(dark_img))
    if cond['LED'] and not cond['Laser']:
        dark_imgs_0608.append(dark_img)

#dark_imgs_0609 = #glob.glob()
blank_img_0608 = np.load('blank_0608.npy')#get_average_blue_img(dark_imgs_0608)
#dark_img_0609 = get_average_blue_img(dir_imgs_0609)

# testing purpose
yaml_path = f'../data/yaml/test.yaml'

for live_img_dir in live_img_dirs: 
    dir_name = os.path.basename(live_img_dir)
    cond = simplify_dir_name(dir_name)
    wanted_frame = get_wanted_frames_for_condition(cond, yaml_path)
    print(wanted_frame)
    live_img_paths = list(Path(live_img_dir).glob('*.png'))
    if on_0608(os.path.basename(live_img_dir)):
        live_im = preprocess(live_img_paths, blank_img_0608,
                             wanted_frame, bypass=True)
    else:
        live_im = preprocess(live_img_paths, blank_img_0609,
                             wanted_frame, bypass=True)

# Storing
