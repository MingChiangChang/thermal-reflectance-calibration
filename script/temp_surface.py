''' Process npys '''
from pathlib import Path
import os
import sys

from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../src/')
from temp_calibration import fit_xy_to_z_surface_with_func
from error_funcs import twod_surface

def parse_npy_fn(npy_file_name):
    a = npy_file_name.split('_')
    dwell = a[0][:a[0].index('us')]
    power = a[1][:a[1].index('W')]
    return dwell, power

home = Path.home()
path = home / 'Desktop' / 'github' / 'thermal-reflectance-calibration'
path = path / 'data' / 'npy' / 'pfits'

npy_ls = sorted(list(path.glob('*')))

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
    proper_t = t[np.logical_and(t<1000, t>0)]
    temp = np.mean(proper_t)
    std = np.std(proper_t)
    r = [dwell, power, temp, std]
    r = map(float, r)
    result.append(list(r))

result = np.array(result)
result[:,2] = result[:,2]/0.6
pfit, pcov, infodict = fit_xy_to_z_surface_with_func(np.log10(result[:,0]),
                              result[:,1],
                              result[:,2],
                              twod_surface, [1,1,1,1,1,1])
fit_func = twod_surface(*pfit)
x = np.linspace(2.5, 4, 10)
y = np.linspace(20, 110, 10)
xx, yy = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(np.log10(result[:,0]), result[:,1], result[:,2])
dd = np.array([2000, 5000, 10000])
power = np.array([89, 67, 61])
temp = np.array([1260, 1260, 1260])

def f(kappa):
    return fit_func(np.log10(dd), power)/kappa - temp

pfit, pcov, infodict, errmsg, success = leastsq(f, [1],
                                          full_output=1)
print(pfit)
ax.scatter(np.log10(dd), power, temp)
ax.plot_surface(xx, yy, fit_func(xx, yy)/pfit[0], alpha=0.3)
ax.set_xlabel('log dwell')
ax.set_ylabel('Current (amps)')
ax.set_zlabel('a.u.')
ax.set_title('Tpeak')
plt.show()

#result = np.array(result)
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(np.log10(result[:,0]), result[:,1], result[:,3])
#ax.set_xlabel('log dwell')
#ax.set_ylabel('Current (amps)')
#ax.set_zlabel('a.u.')
#ax.set_title('Std')
#plt.show()
