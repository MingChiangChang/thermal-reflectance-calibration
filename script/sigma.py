''' Fit sigma surface '''
from pathlib import Path
from inspect import getfullargspec
import os
import sys

from tqdm import tqdm
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import dill

sys.path.insert(0, '../src/')
from temp_calibration import fit_xy_to_z_surface_with_func
from error_funcs import temp_surface, twod_surface

dill.settings['recurse'] = True

def parse_npy_fn(npy_file_name):
    ''' parse npy file name into dwell and power'''
    a = npy_file_name.split('_')
    dwell = a[0][:a[0].index('us')]
    power = a[1][:a[1].index('W')]
    return dwell, power

def remove_outliers(data):
    ''' Remove outlier with 1.5 IQR and must > 0'''
    a = sorted(data)
    num = len(a)
    IQR = a[(num//4)*3] - a[num//4] # pylint: disable=C0103
    median = a[num//2]
    a = np.array(a)
    return a[np.logical_and(np.logical_and(np.logical_and(a < median + 1.5*IQR,
                            a > median - 1.5*IQR),
                            a>150), a<270)]

fit_surface = fit_xy_to_z_surface_with_func
surface_func = twod_surface
guess = [1 for _ in range(len(getfullargspec(surface_func).args))]

home = Path.home()
path = home / 'Desktop' / 'github' / 'thermal-reflectance-calibration'
path = path / 'data' / 'npy' / 'pfits'

npy_ls = sorted(list(path.glob('*')))

################# Load Data ##################
result = []
for npy in npy_ls:
    npy_fn = os.path.basename(npy)
    try:
        pfit = np.load(npy)
    except ValueError:
        print(f"{npy_fn} cannot be read.")

    npy = os.path.basename(npy)
    dwell, power = parse_npy_fn(str(npy_fn))

    sigma = pfit[:,:,4]
    proper_sigma = remove_outliers(np.ravel(sigma))
    sigma = np.mean(proper_sigma)
    std = np.std(proper_sigma)
    r = [dwell, power, sigma, std]
    r = map(float, r)
    result.append(list(r))

result = np.array(result)

pfit, pcov, infodict = fit_surface(np.log10(result[:,0]),
                                   result[:,1],
                                   result[:,2],
                                   surface_func, guess,
                                   uncertainty=result[:,3])
fit_func = surface_func(*pfit)
print(pcov)
x = np.linspace(2.5, 4, 10)
y = np.linspace(20, 110, 10)
xx, yy = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cmap = ListedColormap(['r', 'g', 'b'])
#ax.scatter(np.log10(dd), pp, si_melt_temp, c=si_melt_temp, cmap=cmap)
ax.scatter(np.log10(result[:,0]), result[:,1], result[:,2],
           c=result[:,2], cmap='bwr', s=40)
for i in range(result.shape[0]):
    ax.plot([np.log10(result[i,0]), np.log10(result[i, 0])],
            [result[i,1], result[i,1]], 
            [result[i,2]-result[i,3], result[i, 2]+result[i,3]], c='r') 
ax.plot_surface(xx, yy, fit_func(xx, yy), alpha=0.3)
ax.set_xlabel('log dwell')
ax.set_ylabel('Current (amps)')
ax.set_zlabel('a.u.')
ax.set_title('Std')
plt.show()


