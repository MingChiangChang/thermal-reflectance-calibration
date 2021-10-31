''' Script for analyze even temperature test '''
from os.path import basename
from pathlib import Path
from math import ceil
import glob
import sys
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '../src')
from preprocess import parse_laser_condition, parse_name, recon_fn

def existed_conditions(path):
    png_ls = glob.glob(path + '*')
    ecs = set()
    print(png_ls)
    for png in png_ls:
        print(png)
        condition = basename(png)
        condition = condition.split('_')
        cond = (int(condition[0][:condition[0].index('us')]),
             int(condition[1][:2]))
        ecs.add(cond)
    return ecs

if __name__ == '__main__':
    FIG_PER_ROW = 6

    home = Path.home()
    data_path = home / 'Desktop' / 'Data' 
    d = 'at_CHESS_LD/' # Make this parsed argument

    dir_path = data_path / d 
    des_path = data_path / 'summary' / d 
    print(str(des_path))

    try:
        os.mkdir(des_path)
    except FileExistsError:
        print(f'{des_path} directory exists.')
    
    dirs = sorted(list(dir_path.glob('*')))[73:]
    print(dirs)
    print(basename(dirs[0]))

    #ecs = existed_conditions(des_path + '/')
    conditions = [parse_laser_condition(basename(d)) for d in dirs]
    for idx, c in enumerate(conditions):
        print(idx, c)
    # generate one summary graph for each run
    for d, condition in zip(dirs, conditions):
        print(tuple(condition.values()))
        # load data in
        fig_paths = sorted(list(d.glob('*')))
        running_conditions = [parse_name(basename(fig_path))
                                         for fig_path in fig_paths]

        num_sets = len({rc['Run'] for rc in running_conditions})

        for i in range(num_sets):
            a_run = []
            for rc in running_conditions:
                if rc['Run']==i and rc['LED'] and rc['Laser']: # This line is key
                    print(recon_fn(rc))
                    a_run.append(plt.imread(d / recon_fn(rc)).astype(float)[:,:,2])

            rows = ceil(len(a_run)/FIG_PER_ROW)
            fig, axs = plt.subplots(rows, FIG_PER_ROW, figsize=(15, 7), squeeze=True)

            for r in range(rows):
                for c in range(FIG_PER_ROW):
                    frame_num = r*FIG_PER_ROW+c
                    axs[r][c].set_title(str(frame_num))
                    axs[r][c].xaxis.set_visible(False)
                    axs[r][c].yaxis.set_visible(False)
                    if frame_num < len(a_run):
                        axs[r][c].imshow(a_run[frame_num])
            title = '{}us_{}W_Run{}'.format(int(condition['dwell']),
                                            int(condition['power']),
                                            str(i))
            fig.suptitle(title)
            print(str(des_path))
            plt.savefig(str(des_path) + '/' +title + '.png')
            fig.clear()
            plt.close(fig)
