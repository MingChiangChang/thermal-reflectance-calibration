'''
Functions for turning raw data into usable input data for
CalibMnger. The process involves stacking beams on top of
each other and averages.
'''

import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from temp_calibration import fit_center

KAPPA = 1.2*10**-4

def preprocess(live_img_path, wanted_frames, blank_img_input, x_r=(0, 1024), y_r=(0, 1280),
               blank_bypass=False, center_estimate=False,
               t=False, dwell=False, plot=False, savefig=False):
    '''
    Take a directory of live images, a directory of blank images
    and a yaml file as input then output the image that is ready
    to be put into the calibration module. Probably don't want
    to include kappa.
    '''
    if blank_bypass:
        blank_im = blank_img_input[x_r[0]:x_r[1], y_r[0]:y_r[1]]
    else:
        blank_im = get_average_blue_img(blank_img_input)[x_r[0]:x_r[1], y_r[0]:y_r[1]]
    png_ls = generate_png_names_from_dict(wanted_frames)
    png_ls = [live_img_path + '/' + png for png in png_ls]
    live_imgs = read_img_array(png_ls)[:, x_r[0]:x_r[1], y_r[0]:y_r[1]]
    if center_estimate:
        live_imgs, xs, ys, pfits = shift_calibration_to_imgs(live_imgs, blank_im,
                                                      center_estimate,
                                                      t, dwell, plot, savefig)
    else:
        live_imgs, xs, ys, pfits = shift_calibration_to_imgs(live_imgs, blank_im,
                                                    t=t, dwell=dwell, plot=plot, savefig=savefig)
    live_img = np.mean(live_imgs, axis=0)
    return live_img, xs, ys, pfits

def preprocess_by_frame(live_img, blank_img, x_r, y_r):
    live = live_img - blank_img
    R = np.mean(blank_img[300:,:])
    live = (live/KAPPA/R)[x_r[0]:x_r[1], y_r[0]:y_r[1]]
    x, y, pfit = fit_center(live)
    return x, y, pfit

def generate_png_name(run, led, laser, num):
    '''
    Return png names with given run, LED status (bool), Laser status (bool), number of frame
    '''
    l = 'On' if led else 'Off'
    p = 'On' if laser else 'Off'
    return f'Run-{run.zfill(4)}_LED-{l}_Power-{p}_Frame-{num.zfill(4)}.png'

def generate_png_names_from_dict(frame_dict):
    '''
    Generate png names with the proper format from a dictionary that defines
    all the frame numbers
    '''
    png_ls = []
    for run in frame_dict:
        for frame in frame_dict[run]:
            png_ls.append(generate_png_name(str(run), True,
                                            True, str(frame)))
    return png_ls

def read_img_array(img_ls, channel_num=2):
    '''
    Asserted all the images have the same dimesnion and automatically detect the
    dimension of the images. Then read the given channel of the image into
    a np array with corresponding shape.
    '''
    width, height = get_dimension(img_ls[0])
    read_img_arr = np.zeros((len(img_ls), width, height))
    for idx, img in tqdm(enumerate(img_ls), desc='Reading Image Array...'):
        read_img_arr[idx] = plt.imread(img).astype(float)[:,:,channel_num]
    return read_img_arr

def get_dimension(img):
    ''' give the dimension of the 3rd channel of the image (nominally blue channel)'''
    return plt.imread(img).astype(float)[:,:,2].shape

def get_average_blue_img(img_ls):
    ''' Return the average of the third channel fo the given array of image path'''
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

def get_dir_name_from_cond(cond):
    '''
    Get the directory name by the condition in the form {dwell}us_{power}W
    '''
    l = cond.split('_')
    dwell = l[0][:l[0].index('us')]
    power = float(l[1][:l[1].index('W')])
    return f'{dwell.zfill(5)}us_{power:06.2f}W'

def get_calib_dir_name_from_dwell(dwell):
    return f'{str(int(dwell)).zfill(5)}us_000.00W'

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
                   t=False, dwell=False, plot=False, savefig=False):
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
    pfits = []
    for idx, _ in tqdm(enumerate(imgs), desc='Read and find center...'):
        imgs[idx] = (imgs[idx]-blank_im)/blank_im/KAPPA
        if center_estimate:
            x, y, pfit = fit_center(imgs[idx], center_estimate, t, dwell, idx, plot, savefig)
        else:
            x, y, pfit = fit_center(imgs[idx], t=t, dwell=dwell,
                                    num=idx, plot=plot, savefig=savefig)
        xs.append(x)
        ys.append(y)
        pfits.append(pfit)
    xs = np.array(xs)
    ys = np.array(ys)
    pfits = np.array(pfits)
    mx = np.mean(xs)
    my = np.mean(ys)
    for idx, _ in enumerate(imgs):
        imgs[idx] = np.roll(imgs[idx], int(mx-xs[idx]), axis=0)
        imgs[idx] = np.roll(imgs[idx], int(my-ys[idx]), axis=1)
    return imgs, xs, ys, pfits

def im_to_temp(im, blank_im, kappa):
    '''
    Turn reflectance data and turn into temperature
    
    Params:
    im: Input reflectance data
    blank_im: blank image. Should have same dimension with im
    kappa: Constant for increasing temperature to reflectance
    '''
    im = im-blank_im
    im = im/blank_im/kappa
    return im

def plot_blue(png_ls, save_at=None):
    '''
    Plot the third channel of the image and have the option
    to save it at a directory designated by the param 'save_at'
    '''
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
