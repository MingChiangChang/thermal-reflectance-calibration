'''
Functions for turning raw data into usable input data for 
CalibMnger. The process involves stacking beams on top of 
each other and averages.
'''

import glob
import os
import yaml
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from temp_calibration import fit_center

def preprocess(live_img_path, wanted_frames, blank_img_input,
               blank_bypass=False, center_estimate=False):
    '''
    Take a directory of live images, a directory of blank images
    and a yaml file as input then output the image that is ready
    to be put into the calibration module. Probably don't want
    to include kappa.
    '''
    if blank_bypass:
        blank_im = blank_img_input
    else:
        blank_im = get_average_blue_img(blank_img_input)
    png_ls = generate_png_names_from_dict(wanted_frames)  
    png_ls = [live_img_path + '/' + png for png in png_ls]
    live_imgs = read_img_array(png_ls)
    if center_estimate:
        live_imgs, xs, ys = shift_calibration_to_imgs(live_imgs, blank_im, 
                                                      center_estimate)
    else:
        live_imgs, xs, ys  = shift_calibration_to_imgs(live_imgs, blank_im) 
    live_img = np.mean(live_imgs, axis=0)
    return live_img, xs, ys

def generate_png_name(run, led, laser, num):
    l = 'On' if led else 'Off'
    p = 'On' if laser else 'Off'
    return f'Run-{run.zfill(4)}_LED-{l}_Power-{p}_Frame-{num.zfill(4)}.png'

def generate_png_names_from_dict(frame_dict):
    png_ls = []
    for run in frame_dict:
        for frame in frame_dict[run]:
            png_ls.append(generate_png_name(str(run), True,
                                            True, str(frame)))
    return png_ls

def read_img_array(img_ls):
    width, height = get_dimension(img_ls[0])
    read_img_arr = np.zeros((len(img_ls), width, height)) 
    for idx, img in tqdm(enumerate(img_ls), desc=f'Reading Image Array...'):
        read_img_arr[idx] = plt.imread(img).astype(float)[:,:,2]
    return read_img_arr 

def get_dimension(img):
    return plt.imread(img).astype(float)[:,:,2].shape

def get_average_blue_img(img_ls):
    imgs = read_img_array(img_ls)
    return np.mean(imgs, axis=0)
 
def parse_laser_condition(dir_name):
    cond = {}
    cond['dwell'] = int(dir_name[:dir_name.index('us')])
    cond['power'] = float(dir_name[dir_name.index('_')+1:-1])
    return cond

def get_full_condition(cond):
    dwell = str(cond['dwell'])+'us'
    power = str(int(cond['power']))+'W'
    return f'{dwell}_{power}'

def simplify_dir_name(name):
    return get_full_condition(parse_laser_condition(name))

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

def get_highest_power_for_cond(cond, all_conds):
    pws = [c['power'] for c in all_conds 
                      if c['dwell'] == cond['dwell']]
    return np.max(pws)

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

def shift_calibration_to_imgs(imgs, blank_im, center_estimate=False,
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
    xs = []
    ys = []

    for idx, _ in tqdm(enumerate(imgs), desc='Read and find center...'):
        imgs[idx] = imgs[idx]-blank_im
        if center_estimate:
            x, y = fit_center(imgs[idx], center_estimate, t, dwell, num, plot)
        else:
            x, y = fit_center(imgs[idx], t=t, dwell=dwell, num=num, plot=plot)
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    mx = np.mean(xs)
    my = np.mean(ys)
    for idx, _ in enumerate(imgs):
        imgs[idx] = np.roll(imgs[idx], int(mx-xs[idx]), axis=0)
        imgs[idx] = np.roll(imgs[idx], int(my-ys[idx]), axis=1)
    return imgs, xs, ys

def im_to_temp(im, blank_im, kappa):
    im = im-blank_im
    im = im/blank_im/kappa
    return im

def plot_blue(png_ls, save_at=None):
    for png in png_ls:
        sc = plt.imshow(plt.imread(png)[:,:,2])
        plt.colorbar(sc)
        if save_at is not None:
            plt.savefig(f'{save_at}/{os.path.basename(png)}')
        else:
            plt.show()
        plt.close()

def plot_blue_for_dir(d, filetype):
    '''
    Take a directory, find all the {filetype} in the directory
    and replot it with only the blue channel and save at 
    the same directory
    '''
    p = Path(d)
    img_ls = p.glob(f'*.{filetype}')
    plot_blue(img_ls, save_at=str(p))

def get_wanted_frames_for_condition(condition, yaml_dict):
    '''
    yaml structure:
    {cond: run_num: [wanted frames]}
    '''
    try:
        for run in yaml_dict[condition]:
            l = yaml_dict[condition][run]
            yaml_dict[condition][run] = make_continuous(l)
        return yaml_dict[condition] 
    except KeyError:
        print('Condition not found! Returning empty list.')
        return []

def make_continuous(l):
    '''
    Take a list with numbers, return a continuous integer
    list between the minimum and the maximum of the list
    e.g. [1,5] -> [1,2,3,4,5]
    '''
    minimum = np.min(l) 
    maximum = np.max(l)
    return np.linspace(minimum, maximum, maximum-minimum+1).astype(int).tolist()

if __name__ == '__main__':
    # For testing purpose
    dir_path = '/Users/mingchiang/Desktop/Work/sara-socket-client/Scripts/0531_test/00500us_020.00W/'

    png_ls = sorted(glob.glob(dir_path + '*'))
