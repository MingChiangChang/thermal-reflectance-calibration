import glob

import numpy as np
import matplotlib.pyplot as plt

import preprocess as pp

dir_path = '/Users/mingchiang/Desktop/Work/sara-socket-client/Scripts/0531_test/00500us_020.00W/'

png_ls = sorted(glob.glob(dir_path + '*'))

condition_ls = pp.parse_names(png_ls)

live = []
dark = []
blank = []
dark_blank = []

for path, cond in zip(png_ls, condition_ls):
    print(cond)
    if cond['LED'] and cond['Laser']:
        live.append(dict(cond, path=path))
    elif cond['LED']:
        blank.append(dict(cond, path=path))
    elif cond['Laser']:
        dark.append(dict(cond, path=path))
    else:
        dark_blank.append(dict(cond, path=path))

print(len(live), len(dark), len(blank), len(dark_blank))


