import os
import math as mt
import cv2
import numpy as np
from numpy import pi, r_
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import scipy.linalg as la
import glob
import copy as cp
from mpl_toolkits.mplot3d import Axes3D
from skimage.restoration import estimate_sigma
from scipy.ndimage import  rotate
import yaml

def fit_center(data, center_estimate=False, t=False, dwell=False, num=False, plot=False, savefig=False):
    if center_estimate:
        params = list(moments(data))
        params[1] = center_estimate[0]
        params[2] = center_estimate[1]
        pfit, s_sq = fit_with(gaussian_shift, data, param=params)
    else:
        pfit, s_sq = fit_with(gaussian_shift, data, param_estimator=moments)
    fit = gaussian_shift(*pfit)
    xs, ys = np.indices(data.shape)
    fitted = fit(*(xs, ys))
    tpeak = np.max(fitted)
    center = np.where(fitted==tpeak)
    print(center)
    xs = center[0][0]
    ys = center[1][0]
    if plot:
        fig, axs = plt.subplots(4)
        axs[0].imshow(data)
        axs[1].imshow(fitted)
        axs[2].plot(fitted[xs])
        axs[2].plot(data[xs])
        axs[2].set_title('x fit')
        axs[3].plot(fitted[:, ys])
        axs[3].plot(data[:, ys])
        axs[3].set_title('y_fit')
        #plt.show()
        if savefig:
            plt.savefig(f'{t}_{dwell}_{num}.png')
        else:
            plt.show()
        fig.clear()
        plt.close()
    return xs, ys


def fit_xy_to_z_surface_with_func(x, y, z, func, param, uncertainty=None):
    if uncertainty is None:
        error_func = lambda p: np.ravel(func(*p)(x, y) - z)
    else:
        error_func = lambda p: np.ravel((func(*p)(x, y) - z)/uncertainty)

    pfit, pcov, infodict, errmsg, success = leastsq(error_func, param,
                                          full_output=1)

    if success not in [1,2,3,4]:
        print('Fitting Failed!')
        print('Error Message: {}'.format(errmsg))

    return pfit, pcov, infodict

def read_and_add_weight_to_images(fp_ls):
    for i, fp in enumerate(fp_ls):
        print(fp)
        if i==0:
            dst = plt.imread(fp)
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(plt.imread(fp), alpha, dst, beta, 0.0)
    return dst

def fit_mask(data, fit, thresh):
    fit_data = fit(*np.indices(data.shape))
    mask = fit_data < thresh
    ay = np.where(np.any(~mask, axis=1))
    ax = np.where(np.any(~mask, axis=0))
    if len(ax[0]) > 1:
        dx = np.max(ax) - np.min(ax)
    else:
        dx = 0.
    print("DX", dx)
    return mask, dx*0.5

def trail(data, u = 100., v = 100., b = 100, direction='Down'):
    X, Y = np.indices(data.shape)
    if direction == 'Down':
        mask = np.logical_or((Y - v)**2 > b**2,  X < u)
    else:
        mask = np.logical_or((Y - v)**2 > b**2,  X > u)
    return mask

def ellipse(data, u = 100., v = 100., a = 100, b = 50):
    X, Y = np.indices(data.shape)
    mask = (X - u)**2/a**2 + (Y - v)**2/b**2 > 1.
    return mask

def estimate_noise(img):
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(img,-1,kernel)
    blur = cv2.blur(img,(3,3))
    diff = np.abs(np.array(img).astype(float) - np.array(blur).astype(float))
    return np.sum(diff)/img.shape[0]/img.shape[1]
    #return estimate_sigma(img, multichannel=False, average_sigmas=False)

def surface_plot (matrix, fit, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    x, y = np.meshgrid(range(matrix.shape[0]), range(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf0 = ax.plot_surface(x.T, y.T, np.array(matrix), alpha=0.5, cmap=plt.cm.copper)
    surf1 = ax.plot_surface(x.T, y.T, fit(*np.indices(matrix.shape)), **kwargs)
    return (fig, ax, surf1)


def gaussian(height, center_x, center_y, width_x, width_y, rho):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp( -(
        ((center_x-x)/width_x)**2 +
        ((center_y-y)/width_y)**2 -
        (2*rho*(x - center_x)*(y - center_y))
        /(width_x*width_y))/(2*(1-rho**2)))

def gaussian_shift(height, center_x, center_y, width_x, width_y, rho, shift):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp( -(
        (abs(center_x-x)/width_x)**2 +
        (abs(center_y-y)/width_y)**2 -
        (2*rho*(x - center_x)*(y - center_y))
        /(width_x*width_y))/(2*(1-rho**2))) + shift

def g_gaussian(height, center_x, center_y, width_x, width_y, eg):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp( -(
        (np.abs(center_x-x)/width_x)**eg +
        (np.abs(center_y-y)/width_y)**2
        )/np.sqrt(2.*eg))

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/data.sum()
    y = (Y*data).sum()/data.sum()
    
    if x > data.shape[0] or x<0:
        x = data.shape[0]/2
    if y > data.shape[1] or y<0:
        y = data.shape[1]/2

    col = data[:, int(y)]
    #print(np.abs((np.arange(col.size)-x)**2*col))
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/np.abs(col.sum()))
    row = data[int(x), :]
    #print(np.abs((np.arange(row.size)-y)**2*row))
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/np.abs(row.sum()))
    height = data.max()
    rho = 0.0
    shift = 0.0
    #print('Moments: {}, {}, {}, {}, {}'.format(height, x, y, width_x, width_y))
    return height, x, y, width_x, width_y, rho, shift

def edgeworth(x, x0, s, sk, ku):
    return 1/(2*np.pi*s)\
           * np.exp(-(x-x0)**2/(2*s**2))\
           * edge_expansion( (x-x0)/s, sk, ku) 

def edge_expansion(r, k3, k4):
    return 1 + k3*(r**3-3*r)/6 + k4/12*(r**4-6*r**2+3)\
           + k3**2*(r**6-15*r**4+45*r**2-15)/72

def twod_edgeworth(height, x0, y0, s_x, s_y, sk_x, sk_y, ku_x, ku_y):
    return lambda x, y: (height 
                        * edgeworth(x, x0, s_x, sk_x, ku_x)
                        * edgeworth(y, y0, s_y, sk_y, ku_y))

def default_value_approx(data):
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    #print(np.abs((np.arange(col.size)-x)**2*col))
    s_x0 = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/np.abs(col.sum()))
    sk_x0 = ((np.arange(col.size)-x)**3*col).sum()/s_x0**3
    ku_x0 = ((np.arange(col.size)-x)**4*col).sum()/s_x0**4-3
    row = data[int(x), :]
    #print(np.abs((np.arange(row.size)-y)**2*row))
    s_y0 = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/np.abs(row.sum()))
    sk_y0 = ((np.arange(row.size)-y)**3*row).sum()/s_y0**3
    ku_y0 = ((np.arange(row.size)-y)**4*row).sum()/s_y0**4-3
    height = data.max()
    print((f"Default height: {height}, x: {x}, y: {y}, s_x: {s_x0}, "
           f"s_y: {s_y0}, sk_x: {sk_x0}, sk_y: {sk_y0}, ku_x: {ku_x0}, "
           f"ku_y: {ku_y0}"))
    return height, x, y, s_x0, s_y0, sk_x0, sk_y0, ku_x0, ku_y0 

def fit_with(func, data, mask=None, param_estimator=None, param=None, maxfev=1000):
    if param is None:
        param = param_estimator(data)
    if mask is None:
        errorfunction = lambda p: np.ravel(func(*p)(*np.indices(data.shape)) - data)
    else:
        X, Y = np.indices(data.shape)
        A = np.c_[X[mask], Y[mask]].T
        errorfunction = lambda p: np.ravel(func(*p)(*A) - data[mask])
    
    pfit, pcov, infodict, errmsg, success = leastsq(errorfunction, param, full_output=1, maxfev=maxfev)
    
    SUCCESS_STATE = [1,2,3,4]
    if success in SUCCESS_STATE:
        print('Fitting is successful.')
        #print(pfit)
    else:
        print('Fitting failed.')
        print('Reason: ', errmsg)
    s_sq = (errorfunction(pfit)**2).sum()/(len(data.flatten()))
    #pcov = pcov * s_sq
    error = []
    for i in range(len(pfit)):
        try:
            pcov = pcov * s_sq
            error.append(np.absolute(pcov[i][i])**0.5)
        except:
            error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error)
    return pfit_leastsq, np.sqrt(s_sq)

def get_box(profile, threshold=0.6):
    mask = profile>np.max(profile)*threshold
    ay = np.where(np.any(mask, axis=0))
    ax = np.where(np.any(mask, axis=1))
    x_center = np.mean(ax)
    y_center = np.mean(ay)
    area = np.sum(mask)
    x_r = np.sqrt(area/(3*np.pi))
    y_r = 3*x_r
    x_shift = int(x_center-2.5*x_r) if x_center-3*x_r > 0 else 0
    x_width = int(4*x_r)
    y_shift = int(y_center-2*y_r) if y_center-2*y_r > 0 else 0
    y_width = int(4*y_r) 
    return x_shift, x_width, y_shift, y_width 
