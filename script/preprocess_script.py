'''
Script for preprocessing images for temperature surface fitting
Take raw reflectance and yaml file and output npy files
Each image is fitted and shifted in order to be properly stacked.
'''
from pathlib import Path
import sys
import yaml
import json

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../src')
from preprocess import parse_laser_condition
from preprocess import get_wanted_frames_for_condition, preprocess
from preprocess import get_highest_power_for_cond
from preprocess import get_dir_name_from_cond
from preprocess import generate_png_name, read_img_array
from preprocess import shift_calibration_to_imgs, preprocess_by_frame
from preprocess import parrallel_processing_frames
from temp_calibration import self_blank

### Global
x_r = (300, 650)
y_r = (0, 850)
mask = np.zeros((1024, 1280))
mask[:, 1000:] = 1
mask[800:, :] = 1
mask = mask.astype(bool)

# Linux
home = Path.home()
PATH = home / 'Desktop' / 'Data' / 'Chess_diode'
YAML_PATH = '../data/yaml/Chess_diode.yaml'

with open(YAML_PATH, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
print(yaml_dict)

live_img_conds = list(yaml_dict.keys())

for idx, p in enumerate(live_img_conds):
    print(idx, p)

all_conds = [parse_laser_condition(d) for d in live_img_conds]
print(all_conds)

for cond in live_img_conds[75:]:
    print(f'Working on {cond}')
    dir_name = get_dir_name_from_cond(cond)
    cond_dict = parse_laser_condition(dir_name)
    wanted_frame = get_wanted_frames_for_condition(cond, yaml_dict)
    #max_pw = int(get_highest_power_for_cond(cond_dict, all_conds))
    dw = cond_dict['dwell']
    blank = np.load(f'../data/npy/chess_blanks/{dw}.npy')
    pfit_ls = [] 
    
    for run in tqdm(wanted_frame):
        png_ls = [PATH / dir_name / generate_png_name(str(run), True, True, str(i))
                      for i in wanted_frame[run]]
        live_imgs = np.zeros((len(png_ls), 1024, 1280))
        blank_imgs = np.zeros((len(png_ls), 1024, 1280))
        for idx, png in enumerate(png_ls):
            live_imgs[idx] = plt.imread(png)[:,:,2]
            blank_imgs[idx] = self_blank(live_imgs[idx], blank, mask)

        temp_fit = parrallel_processing_frames(live_imgs, blank_imgs, x_r, y_r)
        for fit in temp_fit:
            pfit_ls.append(fit)
        
    np.save(f'../data/npy/chess_TR/{cond}_pfit.npy', np.array(pfit_ls))
