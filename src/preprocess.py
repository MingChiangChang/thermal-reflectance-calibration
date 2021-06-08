'''
Functions for turning raw data into usable input data for 
CalibMnger. The process involves stacking beams on top of 
each other and averages.
'''

import glob
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from temp_calibration import fit_center

def parse_laser_cond(dir_name):
    cond = {}
    cond['dwell'] = int(dir_name[:dir_name.index('us')])
    cond['power'] = float(dir_name[dir_name.index('_')+1:-1])
    return cond

def get_full_condition(cond):
    dwell = str(cond['dwell'])+'us'
    power = str(int(cond['power']))+'W'
    return f'{dwell}_{power}'

def parse_names(png_ls):
    cond_ls = []
    for png in png_ls:
        fn = os.path.basename(png)
        cond_ls.append(parse_name(fn))
    return cond_ls
    
def parse_name(fn):
    temp = [cond.split('-')[1] for cond in fn.split('_')]
    temp[0] = int(temp[0])
    temp[1] = True if temp[1]=='On' else False
    temp[2] = True if temp[2]=='On' else False
    temp[3] = int(temp[3][:4])
    d = {'Run': temp[0],
         'LED': temp[1],
         'Laser': temp[2],
         'num': temp[3]}
    return d

def recon_fn(name_dict):
    LED_status = 'On' if name_dict['LED'] else 'Off'
    Laser_status = 'On' if name_dict['Laser'] else 'Off'

    return 'Run-{}_LED-{}_Power-{}_Frame-{}.png'.format(
                str(name_dict['Run']).zfill(4),
                LED_status, Laser_status, 
                str(name_dict['num']).zfill(4))


def average_images(png_ls):
    im_ls = []
    for png in tqdm(png_ls, desc='Reading imgs...'):
        im_ls.append(plt.imread(png).astype(float)[:,:,2])
    im_arr = np.array(im_ls)
    return np.mean(im_arr, axis=0)

def shift_calibration_to_imgs(ims, blank_im, kappa, 
                   t=False, dwell=False, num=False, plot=False):
    '''
    Take an array of images (which are np arrays) and blank image for 
    subtracting the background. The image is then fitted with a double
    gaussian and use the center to algn all the images to get a good 
    profile.

    Input:
    ims: array of images, np arrays
    blank_im: single 2d array
    kappa: thermal reflectance constant calibrated to silicon melt
    # Plotting paramters
    t: tpeak
    dwell: as is
    num: number of frames
    plot: boolean for whether to plot the fitted beam profile
    '''
    im_ls = [] 
    xs = []
    ys = []
    for im in tqdm(ims, desc='Read and find center...'):
        temp = im_to_temp(im, blank_im, kappa)
        im_ls.append(temp)
        x, y = fit_center(temp, t, dwell, num, plot)
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    mx = np.mean(xs)
    my = np.mean(ys)
    for idx, _ in enumerate(im_ls):
        im_ls[idx] = np.roll(im_ls[idx], int(mx-xs[idx]), axis=0)
        im_ls[idx] = np.roll(im_ls[idx], int(my-ys[idx]), axis=1)

    return im_ls

def im_to_temp(im, blank_im, kappa):
    im = im-blank_im
    im = im/blank_im/kappa
    return im

def plot_blue(png_ls, safe_at=None):
    for png in png_ls:
        sc = plt.imshow(plt.imread(png)[:,:,2])
        plt.colorbar(sc)
        if safe_at is not None:
            plt.savefig(f'{safe_at}/{os.path.basename(png)}')
        else:
            plt.show()
        plt.close()

def get_wanted_frames(yaml_path):
    '''
    yaml structure:
    {cond: [wanted frames]}
    '''
    with open(yaml_path, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_dict 

if __name__ == '__main__':
    # For testing purpose
    dir_path = '/Users/mingchiang/Desktop/Work/sara-socket-client/Scripts/0531_test/00500us_020.00W/'

    png_ls = sorted(glob.glob(dir_path + '*'))
