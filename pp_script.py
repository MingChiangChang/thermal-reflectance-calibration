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
dir_path_ls = glob.glob('/Users/mingchiang/Desktop/Work/sara-socket-client/Scripts/0603/*')

output_data = []

# load blank
blank_dir = ('/Users/mingchiang/Desktop/Work/sara-socket-client/'
             'Scripts/TEST/00800us_030.00W/')
blank_path_ls = glob.glob(blank_dir+'Run-0000_LED-On_Power-Off_Frame-*')
blank_imgs = np.zeros((len(blank_path_ls), 1024, 1280))

for idx, blank_path in enumerate(blank_path_ls):
    blank_imgs[idx] = plt.imread(blank_path).astype(float)[:,:,2]

blank_im = np.mean(blank_imgs, axis=0)
plt.imshow(blank_im)
plt.title('Blank')
plt.show()

# Read from yaml
frame_dict = 0

for dir_path in dir_path_ls[10:11]:
    print(dir_path)
    laser_cond = pp.parse_laser_cond(os.path.basename(dir_path))
    png_ls = sorted(glob.glob(dir_path + '/*'))
    condition_ls = pp.parse_names(png_ls)

    live = []
    #dark = []
    #blank = []
    #dark_blank = []

    for path, cond in zip(png_ls, condition_ls):
        print(cond)
        if cond['LED'] and cond['Laser'] 
           and cond['num'] in frame_dict[laser_cond]:
#            # Choose desired frames here
#            # Need a quick visualize method
#            # if cond['num'] in [########]: 
            live.append(dict(cond, path=path))
        #elif cond['LED']:
        #    blank.append(dict(cond, path=path))
        #elif cond['Laser']:
        #    dark.append(dict(cond, path=path))
        #else:
        #    dark_blank.append(dict(cond, path=path))
#
#    print(len(live), len(dark), len(blank), len(dark_blank))
#
#    dark_im = pp.average_images([d['path'] for d in dark])
#    blank_im = pp.average_images([b['path'] for b in blank])
#    db_im = pp.average_images([db['path'] for db in blank])
#
#    # Once usable frame is chosen, just need to shift and stack the images
    live_ims = []
    for l in tqdm(live):
        live_ims.append(plt.imread(l['path']).astype(float)[:,:,2])
    shifted_ims = np.array(pp.shift_calibration_to_imgs(live_ims, blank_im, kappa))
    temp = np.sum(shifted_ims, axis=0)
    if plot:
        plt.imshow(temp)
        plt.title(f'{laser_cond}_avg.png')
        plt.show()
    output_data.append(dict(laser_cond, temp=temp.tolist()))

with open('output.json', 'w') as f:
    json.dump(output_data, f)

