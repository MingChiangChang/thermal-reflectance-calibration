''' A script to quickly generate figures for human to
    choose which live/dark image is good'''

import matplotlib.pyplot as plt
from imageio import imread
import glob
from tqdm import tqdm

path = '/Users/mingchiang/Desktop/2021.03.09_calib/'

dw_num = [250, 500, 1000, 2000, 5000, 10000]
#dw_num = [2000]
dw_str = [str(d)+'us' for d in dw_num]
pw_str = [str(15+5*x)+'W' for x in range(14) ]

print(dw_str, pw_str)
with tqdm(ncols=70, desc='collecting',
          total=len(dw_str)*len(pw_str)) as pbar:
    for dw in dw_str:
        for pw in pw_str:

            live_bmp_paths = sorted(glob.glob(path+dw+'/'+pw+'/live*.bmp'))
            dark_bmp_paths = sorted(glob.glob(path+dw+'/'+pw+'/dark*.bmp'))
            
            live_bmps = []
            dark_bmps = []

            for live_bmp_path in live_bmp_paths:
                live_bmps.append(plt.imread(live_bmp_path))

            for dark_bmp_path in dark_bmp_paths:
                dark_bmps.append(plt.imread(dark_bmp_path))

            fig, axs = plt.subplots(4, 6, figsize=(15,7))
            fig.suptitle(dw+' '+pw, fontsize=16)
            try:
                for i in range(4):
                    for j in range(6):
                        axs[i][j].set_title('live_{}'.format(str(6*i+j)))
                        axs[i][j].imshow(live_bmps[6*i+j][:,:,2])
            except IndexError:
                print(f'{dw} {pw} does not have enough live images!')
            #try:
            #    for i in range(4):
            #        for j in range(3):
            #            axs[i][j+3].set_title('dark_{}'.format(str(3*i+j)))
            #            axs[i][j+3].imshow(dark_bmps[3*i+j])
            #except IndexError:
            #    print(f'{dw} {pw} does not have enough dark images!')
            for i in range(4):
                for j in range(6):
                    axs[i][j].get_xaxis().set_visible(False)
                    axs[i][j].get_yaxis().set_visible(False)
            plt.savefig(path+dw+'_'+pw+'_2.png')
            plt.close()
            pbar.update(1)
