import glob
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import preprocess as pp

kappa = 1.2e-3
plot = True
# Should use some glob to get list of directory path 
#dir_path_ls = sorted(glob.glob('/Users/mingchiang/Desktop/github/thermal-reflectance-calibration/data/0607/*'))
dir_path_ls = ['/Users/mingchiang/Desktop/github/thermal-reflectance-calibration/data/00855us_060.00W/']

for idx, d in enumerate(dir_path_ls):
    print(idx, d)
output_data = []

# load blank
blank_dir = ('/Users/mingchiang/Desktop/github/thermal-reflectance-calibration/data/Calibration_0608/10000us_000.00W/')
blank_path_ls = glob.glob(blank_dir+'Run-0000_LED-On_Power-Off_Frame-*')
blank_imgs = np.zeros((len(blank_path_ls), 1024, 1280))

for idx, blank_path in enumerate(blank_path_ls):
    blank_imgs[idx] = plt.imread(blank_path).astype(float)[:,:,2]

blank_im = np.mean(blank_imgs, axis=0)
plt.imshow(blank_im)
plt.title('Blank')
plt.show()

# Read from yaml
yaml_path = '../data/yaml/test.yaml'
wanted_frames = pp.get_wanted_frames(yaml_path) 

for dir_path in dir_path_ls:
    print(dir_path)
    laser_cond_dict = pp.parse_laser_cond(os.path.basename(dir_path))
    full_condition = pp.get_full_condition(laser_cond_dict)
    print(full_condition)
    png_ls = sorted(glob.glob(dir_path + '/*'))
    condition_ls = pp.parse_names(png_ls)

    live = []

    for path, cond in zip(png_ls, condition_ls):
        print(cond)
        if cond['LED'] and cond['Laser']\
           and cond['num'] in wanted_frames[full_condition][cond['Run']]:
            live.append(dict(cond, path=path))

    # Once usable frame is chosen, just need to shift and stack the images
    live_ims = []
    for l in tqdm(live):
        live_ims.append(plt.imread(l['path']).astype(float)[:,:,2])
    shifted_ims = np.array(pp.shift_calibration_to_imgs(
                           live_ims, blank_im,
                           kappa, laser_cond_dict['power'],
                           laser_cond_dict['dwell'], cond['num'], 
                           plot=True ))
    temp = np.sum(shifted_ims, axis=0)
    if plot:
        plt.imshow(temp)
        plt.title(f'{laser_cond}_avg.png')
        plt.show()
    output_data.append(dict(laser_cond, temp=temp.tolist()))

with open('output.json', 'w') as f:
    json.dump(output_data, f)

