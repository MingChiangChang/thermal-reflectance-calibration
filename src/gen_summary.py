import glob
import os
from os.path import basename
import argparse
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess import parse_laser_condition, parse_name, recon_fn

def existed_conditions(d):
    png_ls = glob.glob(d + '*')
    ecs = set() 
    print(png_ls)
    for png in png_ls:
        print(png) 
        condition = os.path.basename(png)
        condition = condition.split('_')
        c = (int(condition[0][:condition[0].index('us')]),  int(condition[1][:2]))
        ecs.add(c)
    return ecs

if __name__ == '__main__':
    FIG_PER_ROW = 6
    g = '/home/mingchiang/Desktop/Data/'
    dir_path = f'{g}/Calibration_0618/'
    des_path = f'{g}summary/{basename(dir_path)}'
    try:
        os.mkdir(des_path)
    except:
        print(f'{des_path} directory exists.')
    dirs = glob.glob(dir_path + '*')
    print(dirs)
    print(basename(dirs[0]))
    
    #ecs = existed_conditions(des_path + '/')

    conditions = [parse_laser_condition(basename(d)) for d in dirs]

    # generate one summary graph for each run
    for d, condition in tqdm(zip(dirs, conditions)):
        print(tuple(condition.values()))
        # load data in
        fig_paths = sorted(glob.glob(d+'/*'))
        running_conditions = [parse_name(basename(fig_path))
                                         for fig_path in fig_paths] 
        
        num_sets = len(set([rc['Run']for rc in running_conditions]))
        
        for i in range(num_sets):
            a_run = []
            for rc in tqdm(running_conditions, desc="Reading.."):
                if rc['Run']==i and rc['LED'] and not rc['Laser']: # This line is key
                    print(recon_fn(rc))
                    a_run.append(plt.imread(d+'/'+recon_fn(rc)).astype(float)[:,:,2])
     
            rows = ceil(len(a_run)/FIG_PER_ROW)
            print(rows, len(a_run))
            fig, axs = plt.subplots(rows, FIG_PER_ROW, figsize=(15, 7), squeeze=True)

            for r in range(rows):
                for c in range(FIG_PER_ROW):
                    frame_num = r*FIG_PER_ROW+c
                    axs[r][c].set_title(str(frame_num))
                    axs[r][c].xaxis.set_visible(False)
                    axs[r][c].yaxis.set_visible(False)
                    if frame_num < len(a_run):
                        axs[r][c].imshow(a_run[frame_num])
            print(condition)
            title = '{}us_{}W_Run{}'.format(int(condition['dwell']),
                                            int(condition['power']),
                                            str(i))
            fig.suptitle(title)
                                                  
            plt.savefig(des_path + title + '.png')
            fig.clear()
            plt.close(fig)



