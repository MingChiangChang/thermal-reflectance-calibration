import glob
import yaml

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from preprocess import get_wanted_frames_for_condition, get_calib_dir_name_from_dwell
from preprocess import recon_fn

#data = '0618'

#yaml_path = '/home/mingchiang/Desktop/Code/thermal-reflectance-calibration/data/yaml/Calibration_0622.yaml'
yaml_path = '/Users/mingchiang/Desktop/github/thermal-reflectance-calibration/data/yaml/black_blank.yaml'

#dir_path = '/home/mingchiang/Desktop/Data/Calibration_0622/'
dir_path = '/Users/mingchiang/Desktop/Data/Calibration_dot/'

#dwell = ['2924us', '4405us', '6637us', '10000us']
dwell = ['2500us']

for d in dwell:
    
    with open(yaml_path, 'r') as f:
        frames = yaml.load(f, Loader=yaml.FullLoader)

    frames = get_wanted_frames_for_condition(d, frames)
    fs = np.sum([len(frames[i]) for i in frames]) 
    imgs = np.zeros((fs, 1024, 1280))

    n = 0
    for run in tqdm(frames):
        for num in frames[run]:
            cond_dict = {'Run': run,
                 'LED': True,
                 'Laser': False,
                 'num': num}
            fn = recon_fn(cond_dict)
            cond = get_calib_dir_name_from_dwell(d[:d.index('us')])
            imgs[n] = plt.imread(dir_path + cond + '/' + fn).astype(float)[:,:,2]
            n += 1

    #imgs = np.array(imgs)
    blank = np.mean(imgs, axis=0)
    np.save(dir_path+d+'.npy', blank)
    plt.imshow(blank)
    plt.title(str(d))
    plt.show()
