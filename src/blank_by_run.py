import glob
import yaml

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from preprocess import get_wanted_frames_for_condition, get_calib_dir_name_from_dwell
from preprocess import recon_fn

name = 'even_temp_test_calibration_full'

yaml_path = f'../data/yaml/{name}.yaml'

dir_path = f'/Users/mingchiang/Desktop/Data/{name}/'

dwell = ['6637us']

for d in dwell:
    
    with open(yaml_path, 'r') as f:
        frames = yaml.load(f, Loader=yaml.FullLoader)
    print(frames)
    frames = get_wanted_frames_for_condition(d, frames)

    for run in tqdm(frames):
        n = 0
        imgs = np.zeros((len(frames[run]), 1024, 1280)) 
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
        np.save(dir_path+d+f'_{run}.npy', blank)
        #plt.imshow(blank)
        #plt.title(str(d))
        #plt.show()
