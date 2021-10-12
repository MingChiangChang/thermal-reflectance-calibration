import sys
import glob
import yaml

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '../src')
from preprocess import get_wanted_frames_for_condition, get_calib_dir_name_from_dwell
from preprocess import recon_fn

data = '0618'
name = 'even_temp_test_calibration_full'

yaml_path = f'/home/mingchiang/Desktop/github/thermal-reflectance-calibration/data/yaml/even_temp_test.yaml'

dir_path = f'/home/mingchiang/Desktop/Data/{name}/'

dwell = ['6637us_49W']

for d in dwell:
    
    with open(yaml_path, 'r') as f:
        frames = yaml.load(f, Loader=yaml.FullLoader)
    print(frames)
    frames = get_wanted_frames_for_condition(d, frames)
    fs = np.sum([len(frames[i]) for i in frames]) 
    print(fs)
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
