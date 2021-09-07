import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
from functools import partial

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sympy.utilities.lambdify import lambdify
from sympy import Symbol, solve
from scipy.optimize import fsolve, leastsq
from tqdm import tqdm

from CalibMnger import CalibMnger
from error_funcs import twod_surface, power_fit_func, linear
from util import parse

# Mac path
directory = '/Users/mingchiang/Desktop/github/thermal-reflectance-calibration/data/npy/'
# Linux path
directory = '/home/mingchiang/Desktop/Code/thermal-reflectance-calibration/data/npy/'

# Initiate Class
print('Creating Objects...')
img_ls = glob.glob(directory + '*_img.npy')
dw_ls = [parse(os.path.basename(img))[0] for img in img_ls]
pw_ls = [parse(os.path.basename(img))[1] for img in img_ls]
Calib = CalibMnger(img_ls, dw_ls, pw_ls)

Calib.fit_tpeak()
temp = [block.tpeak for block in Calib.block_lst]

# Temperature fitting summary plot
sc = plt.scatter(Calib.dwell_lst, Calib.power_lst, c=temp)
plt.xlabel('log Dwell')
plt.ylabel('Power (W)')
ax = plt.gca()
ax.set_xscale('log')
plt.colorbar(sc)
plt.show()

Calib.store_dw_pw_temp_at('temp.npy')

param_dict = Calib.collect_fitting_params()
mask = ( (param_dict["Std"] >305)
        & (param_dict["Std"] < 598)
        & (np.array(Calib.tpeak_lst) > 50)
        & (np.array(Calib.power_lst) > 30) )
param = (0,0,0,0,0,0)
#Calib.fitting_profile_params(twod_surface, param)
p, pcov, infodict = Calib.get_tpeak_fitting_params(twod_surface, param)
# TODO get jacobian from either manual of sympy
print(pcov.shape)
p.tolist()
t_func = twod_surface(*p)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Calib.power_lst, Calib.dwell_lst, Calib.tpeak_lst)
dw = np.array(Calib.dwell_lst)
pw = np.array(Calib.power_lst)
tp = np.array(Calib.tpeak_lst)
pw_x, dw_y = np.meshgrid(pw, dw)
#ax.plot_surface(pw_x, dw_y, t_func(*(pw_x, dw_y)), facecolor=(0, 1, 0, 0.1))
ax.set_ylabel('log Dwell')
ax.set_xlabel('Power (W)')
ax.set_zlabel('Tpeak (C)')
plt.title('Tpeak fit')
plt.show()


#### Linear Correction ####
real_power_to_melt = {'1000': 72,
                      '2000': 64,
                      '2500': 62,
                      '5000': 55,
                      '6000': 56,
                      '7500': 55,
                      '10000': 53}
#real_power_to_melt = {
#        '567': 69.5,
#        '855': 63,
#        '1288': 59,
#        '1941': 53,
#        '2924': 50.5,
#        '4405': 48.25,
#        '6637': 45.5,
#        '10000': 44 
#        }

fitted_melting_power = {}
lower_melting_power = {}
upper_melting_power = {}
real = []
fitted = []

dwells = np.log10(np.array(list(map(float, real_power_to_melt.keys()))))
powers = np.array(list(map(float, real_power_to_melt.values())))
uncertainty = np.sqrt(np.diag(np.array(Calib.uncertainty_at(powers, dwells))))

for d in real_power_to_melt:
    sol_t = partial(t_func, y=np.log10(float(d)))
    to_fit = lambda x: 1414-sol_t(x)
    fitted_melting_power[d]=fsolve(to_fit, real_power_to_melt[d])[0]

for u, d in zip(uncertainty, real_power_to_melt):
    sol_t = partial(t_func, y=np.log10(float(d)))
    to_fit = lambda x: -u+1414-sol_t(x)
    lower_melting_power[d]=fsolve(to_fit, real_power_to_melt[d])[0]

for u, d in zip(uncertainty, real_power_to_melt):
    sol_t = partial(t_func, y=np.log10(float(d)))
    to_fit = lambda x: u+1414-sol_t(x)
    upper_melting_power[d]=fsolve(to_fit, real_power_to_melt[d])[0]

real = np.array([real_power_to_melt[d] for d in real_power_to_melt])
fitted = np.array([fitted_melting_power[d] for d in real_power_to_melt])
upper = np.array([upper_melting_power[d] for d in real_power_to_melt])
lower =np.array([lower_melting_power[d] for d in real_power_to_melt])
dd = np.log10(np.array([float(key) for key in real_power_to_melt.keys()]))

plt.plot(dd, real, label='Exp', c='b')
plt.errorbar(dd, real, yerr=1, c='b')
plt.plot(dd, fitted, label='Fitted', c='r')
plt.plot(dd, upper, label='Upper bound')
plt.plot(dd, lower, label='Lower bound')
plt.xlabel('log Dwell')
plt.ylabel('Power required to reach 1414C')
plt.legend()
plt.show()

unique_dws = sorted(np.unique(Calib.dwell_lst))

for d in unique_dws:
    # TODO plot uncertainty as well
    # TODO investigate some of the weird fitting
    p_lst, t_lst, uncer_lst = Calib.get_data_along_dw(d)
    p_lst, t_lst, uncer_lst = zip(*sorted(zip(p_lst, t_lst, uncer_lst)))
    xx = np.linspace(np.min(p_lst), np.max(p_lst), 100)
    sol_t = partial(t_func, y=float(d))
    plt.plot(xx, sol_t(xx))
    plt.plot(p_lst, t_lst)
    plt.errorbar(p_lst, t_lst, yerr=np.sqrt(uncer_lst))
    plt.title(str(int(10**d))+' us')
    plt.show()

#k = Calib.get_power_fitting_params(power_fit_func, param)
#k.tolist()
#p_func = power_fit_func(*k)

tpeak = Symbol('tepak', real=True, positive=True)
dwell = Symbol('dwell', real=True, positive=True)
power = Symbol('power', real=True, positive=True)

err_func = lambda p: np.ravel(linear(*p)(fitted)-real)
pfit, _= leastsq(err_func, (1,1))
print(pfit)

linear_corr=linear(*pfit)
p_func = lambdify([tpeak, dwell], solve(tpeak-t_func(linear_corr(power), dwell), power)[1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Calib.tpeak_lst, Calib.dwell_lst, Calib.power_lst)
dw = np.array(Calib.dwell_lst)
pw = np.array(Calib.power_lst)
tp = np.array(Calib.tpeak_lst)
tp_x, dw_y = np.meshgrid(tp, dw)
ax.plot_surface(tp_x, dw_y, p_func(*(tp_x, dw_y)), alpha=0.1)
ax.set_xlabel('Tpeak (C)')
ax.set_ylabel('Dwell (us)')
ax.set_zlabel('Power (W)')
plt.title('Power fit')
plt.show()


# Keys: Height, Std, Base
param_dict = Calib.collect_fitting_params()
print(param_dict)
Calib.fitting_profile_params(twod_surface, param)#, mask)
for param in param_dict:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.array(Calib.power_lst),#[mask], 
               np.array(Calib.dwell_lst),#[mask], 
               param_dict[param])#[mask])
    dw = np.array(Calib.dwell_lst)
    pw = np.array(Calib.power_lst)
    pw_x, dw_y = np.meshgrid(pw, dw)
    fit = twod_surface(*Calib.param_fitting_param[param])
    #ax.plot_surface(pw_x, dw_y, fit(*(pw_x, dw_y)), alpha=0.1)
    ax.set_ylabel('Dwell us')
    ax.set_xlabel('Power (W)')
    ax.set_zlabel('Width in pixels')
    plt.title(param)
    plt.show()
