'''
Script for preprocessing images for temperature surface fitting
Take raw reflectance and yaml file and output npy files
Each image is fitted and shifted in order to be properly stacked.
'''
from pathlib import Path
from multiprocessing import freeze_support
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
from preprocess import parrallel_processing_frames
### Global
x_r = (150, 650)
y_r = (150, 1100)


if __name__ == "__main__":
    freeze_support()
    # Mac
    home = Path.home()
    PATH = home / 'Desktop' / 'Data' / 'even_temp_test' / '06637us_055.00W'
    BLANK_PATH = home / 'Desktop' / 'Data' /\
                 'even_temp_test_calibration_full' / '06637us_000.00W'
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
    #bl = np.load('/home/mingchiang/Desktop/Data/even_temp_test_calibration_full/6637us_49W.npy')
    for cond in yaml_dict:
        dir_name = get_dir_name_from_cond(cond)
        cond_dict = parse_laser_condition(dir_name)
        wanted_frame = get_wanted_frames_for_condition(cond, yaml_dict)
        dw = cond_dict['dwell']
        for run in list(wanted_frame.keys())[:1]:
            png_ls = [PATH / generate_png_name(str(run), True, True, str(i))
                      for i in wanted_frame[run]]
            blank_ls = [BLANK_PATH / generate_png_name(str(run), True, False, str(i))
                      for i in wanted_frame[run]]
            p = np.zeros((len(png_ls), 7))
            
            live_imgs = np.zeros((len(png_ls), 1024, 1280))
            blank_imgs = np.zeros((len(blank_ls), 1024, 1280))
            for idx, png in enumerate(png_ls):
                live_imgs[idx] = plt.imread(png)[:,:,2]
                blank_imgs[idx] = plt.imread(blank_ls[idx])[:,:,2]
                #_, _, pfit = preprocess_by_frame(l, b, x_r, y_r)
                #print(pfit)
                #p[idx] = pfit
            #pfit_ls.append(p)
            t = parrallel_processing_frames(live_imgs, blank_imgs, x_r, y_r)
            print(np.array(t))
    #print(pfit_ls)
