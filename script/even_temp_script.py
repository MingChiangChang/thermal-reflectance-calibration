'''
Script for preprocessing images for temperature surface fitting
Take raw reflectance and yaml file and output npy files
Each image is fitted and shifted in order to be properly stacked.
'''
import yaml
exec(open("insert_path.py").read())

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess import parse_laser_condition
from preprocess import get_wanted_frames_for_condition, preprocess
from preprocess import get_highest_power_for_cond
from preprocess import get_dir_name_from_cond
from preprocess import generate_png_name, read_img_array
from preprocess import shift_calibration_to_imgs, preprocess_by_frame
### Global
x_r = (150, 650)
y_r = (150, 1100)

# Mac
PATH = '/home/mingchiang/Desktop/Data/even_temp_test/06637us_055.00W/'
BLANK_PATH = '/home/mingchiang/Desktop/Data/even_temp_test_calibration_full/06637us_000.00W/'
YAML_PATH = '../data/yaml/even_temp_test.yaml'

with open(YAML_PATH, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
print(yaml_dict)

live_img_conds = list(yaml_dict.keys())

for idx, p in enumerate(live_img_conds):
    print(idx, p)

all_conds = [parse_laser_condition(d)
                   for d in live_img_conds]
intensity = []
pfit_ls = []
bl = np.load('/home/mingchiang/Desktop/Data/even_temp_test_calibration_full/6637us_49W.npy')
for cond in yaml_dict:
    dir_name = get_dir_name_from_cond(cond)
    cond_dict = parse_laser_condition(dir_name)
    wanted_frame = get_wanted_frames_for_condition(cond, yaml_dict)
    dw = cond_dict['dwell']
    for run in wanted_frame:
        png_ls = [PATH + generate_png_name(str(run), True, True, str(i))
                  for i in wanted_frame[run]]
        blank_ls = [BLANK_PATH + generate_png_name(str(run), True, False, str(i))
                  for i in wanted_frame[run]]
        p = np.zeros((len(png_ls), 7))
        for idx, png in enumerate(png_ls):
            b = bl 
            l = plt.imread(png)[:,:,2]
            _, _, pfit = preprocess_by_frame(l, b, x_r, y_r)
            print(pfit)
            p[idx] = pfit
        pfit_ls.append(p)
print(pfit_ls)
#        blank_im = np.load(BLANK_PATH + f'/{dw}us_{run}.npy')
#        blank_im = blank_im[x_r[0]:x_r[1], y_r[0]:y_r[1]]
#        live_imgs = read_img_array(png_ls)[:, x_r[0]:x_r[1], y_r[0]:y_r[1]]
#        res = shift_calibration_to_imgs(live_imgs, blank_im,
#                                        t=cond_dict['power'], dwell=dw,
#                                        plot=True,
#                                        savefig=True)
#        live_im, xs, ys, pfits = res
#        np.save(f'../data/npy/{cond}_{run}_img.npy', live_im)
#        np.save(f'../data/npy/{cond}_{run}_xs.npy', xs)
#        np.save(f'../data/npy/{cond}_{run}_ys.npy', ys)
#        np.save(f'../data/npy/{cond}_{run}_pfit.npy', pfits)
