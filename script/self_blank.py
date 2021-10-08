''' Get blank by live image '''

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../src/')
from temp_calibration import fit_with

home = Path.home()
path = home / 'Desktop' / 'Data' / 'even_temp_test' / '06637us_055.00W'

live = plt.imread(path / 'Run-0002_LED-On_Power-On_Frame-0022.png')[:,:,2]

# load template 
blank = np.load("blank.npy")

mask = np.zeros(live.shape)
mask[-350:,:] = 1
mask[:,-350:] = 1
mask = mask.astype(bool)

def f(a,b,c):
    return lambda x, y: blank[mask] * (a*x + b*y + c)

pfit, err = fit_with(f, live, mask, param=[0 ,0 ,1])
