''' Process npys '''
from pathlib import Path
from inspect import getfullargspec
import os
import sys
import json

from tqdm import tqdm
from scipy.optimize import leastsq
from matplotlib.colors import ListedColormap
from sympy.utilities.lambdify import lambdify
from sympy import Symbol, solve
import numpy as np
import matplotlib.pyplot as plt
import dill

sys.path.insert(0, '../src/')
from temp_calibration import fit_xy_to_z_surface_with_func
from error_funcs import twod_surface, temp_surface

# TODO script for inverse solving

dill.settings['recurse'] = True

############# Function choices ###################
fit_surface = fit_xy_to_z_surface_with_func
surface_func = temp_surface
guess = [1 for _ in range(len(getfullargspec(surface_func).args))]

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
    return a[np.logical_and(np.logical_and(a < median + 1.5*IQR,
                            a > median - 1.5*IQR),
                            a>20)]

############# Loading data ####################
home = Path.home()
path = home / 'Desktop' / 'github' / 'thermal-reflectance-calibration'
path = path / 'data' / 'npy' / 'pfits'

npy_ls = sorted(list(path.glob('*')))

dd = np.array([2000, 5000, 10000])
pp = np.array([88, 66, 61])
si_melt_temp = np.repeat(1414, pp.shape[0])

si = [[2000, 88, 1414, 5],
      [5000, 66, 1414, 5],
      [10000, 61, 1414, 5]]
si_melt = [si for _ in range(3)]
si_melt = np.array(si_melt).reshape((9,4))

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

    t = pfit[:,:,0] + pfit[:,:,-1]
    proper_t = remove_outliers(np.ravel(t))
    temp = np.mean(proper_t)
    std = np.std(proper_t)
    r = [dwell, power, temp, std]
    r = map(float, r)
    result.append(list(r))

result = np.array(result)

pfit, pcov, infodict = fit_surface(np.log10(result[:,0]),
                                   result[:,1],
                                   result[:,2],
                                   surface_func, guess,
                                   uncertainty=result[:,3])
fit_func = surface_func(*pfit)

################### Fitting kappa ##################
def f(scaling): # pylint: disable=C0116
    return fit_func(np.log10(dd), pp)/scaling - si_melt_temp

kappa, pcov, infodict, errmsg, success = leastsq(f, [1],
                                          full_output=1)

################### Plotting ######################
x = np.linspace(2.5, 4, 10)
y = np.linspace(20, 110, 10)
xx, yy = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

cmap = ListedColormap(['r', 'g', 'b'])
ax.scatter(np.log10(dd), pp, si_melt_temp, c=si_melt_temp, cmap=cmap)
ax.scatter(np.log10(result[:,0]), result[:,1], result[:,2]/kappa[0],
           c=result[:,2]/kappa[0], cmap='bwr')
for i in range(result.shape[0]):
    ax.plot([np.log10(result[i,0]), np.log10(result[i, 0])],
            [result[i,1], result[i,1]],
            [(result[i,2]-result[i,3])/kappa[0],
             (result[i, 2]+result[i,3])/kappa[0]], c='purple')
ax.plot_surface(xx, yy, fit_func(xx, yy)/kappa[0], alpha=0.3)
ax.set_xlabel('log dwell')
ax.set_ylabel('Current (amps)')
ax.set_zlabel('a.u.')
ax.set_title('Tpeak')
plt.show()

result[:,2] /= kappa[0]
result = np.concatenate((result, si_melt), axis=0) # Now the melt data is
print(result)                                      # added to the array
                                                   # Be careful!!

######### Iterative process to fit surface and kappa ##########
for i in tqdm(range(10)):
    pfit, pcov, infodict = fit_surface(np.log10(result[:,0]),
                                  result[:,1],
                                  result[:,2],
                                  surface_func, guess, uncertainty=result[:,3])
    fit_func = surface_func(*pfit)

    def f(scaling): # pylint: disable=C0116, E0102
        return fit_func(np.log10(dd), pp)/scaling - si_melt_temp

    kappa, pcov, infodict, errmsg, success = leastsq(f, [1],
                                          full_output=1)
    result[:-3,2] /= kappa[0]

t = (pfit/kappa).tolist()

############### Storing temperauture and power function ################
t_func = surface_func(*t)
with open("../data/t_func.d", "wb") as f:
    dill.dump(t_func, f)

tpeak = Symbol('tpeak', real=True, positive=True)
dwell = Symbol('dwell', real=True, positive=True)
power = Symbol('power', real=True, positive=True)
p_func = lambdify((tpeak, dwell),
                  solve(tpeak-t_func(dwell, power), power)[1],
                  modules = "numpy")
with open("../data/p_func.d", 'wb') as f:
    dill.dump(p_func, f)
