import numpy as np
import matplotlib.pyplot as plt
import cv2 

from os.path import basename
from scipy.optimize import leastsq, least_squares

from error_funcs import twod_triangle, gaussian, edgeworth
from temp_calibration import fit_with, get_box, gaussian_shift, moments

class Block():
    
    kappa = 1*10**-4

    def __init__(self, img, dwell, power):
        
        self.temp = img/self.kappa 
        self.beam = self.temp 

        self.dwell = dwell
        self.power = power

        self.tpeak = 0
        self.center = 0
        self.profile_params = None

        self.x_shift = 0
        self.y_shift = 0
        self.x_width = 0
        self.y_width = 0
    
    def __repr__(self):
        return ('Condition: {}us, {}W\n'.format(self.dwell, self.power)
               +'Fitted results: {}C \n'.format(self.tpeak)
               +'Center: {}\n'.format(self.center) 
               +'Fitting paramters: {}\n'.format(self.profile_params)
               +'Box info: {}, {}, {}, {}\n'.format(self.x_shift,
                                                    self.y_shift,
                                                    self.x_shift+self.x_width,
                                                    self.y_shift+self.y_width))

    def __str__(self):
        return ('Condition: {}us, {}W\n'.format(self.dwell, self.power)
               +'Fitted results: {}C \n'.format(self.tpeak)
               +'Center: {}\n'.format(self.center)
               +'Fitting paramters: {}\n'.format(self.profile_params)
               +'Box info: {}, {}, {}, {}\n'.format(self.x_shift,
                                                    self.y_shift,
                                                    self.x_shift+self.x_width,
                                                    self.y_shift+self.y_width))
    
    #def process_image(self):
    #    '''
    #    Process the loaded files into temperature profile ready to
    #    be fitted.
    #    '''
    #    live_sub = self.live-self.blank
    #    dark_sub = self.dark-self.dark_blank
    #    dI = live_sub-dark_sub
    #    therm = dI/self.blank
    #    self.temp = therm/self.kappa
    #    del self.dark, self.blank, self.dark_blank, self.live

    def get_beam(self, threshold=0.6):
        # Need to be changed
        self.x_shift, self.x_width,\
        self.y_shift, self.y_width = (200, 400, 200, 900) 
        print(self.x_shift, self.x_width, self.y_shift, self.y_width)
        self.beam = self.temp[self.x_shift:self.x_shift+self.x_width,
                              self.y_shift:self.y_shift+self.y_width]

    def fit_center(self):
        '''
        Fitting the center center of the beam using a 2d triangle + 
        gaussian function
        '''
        #def err(peak_x, peak_y, x0, y0, s1_x, s2_x, s1_y, s2_y, height,
        #        center_x, center_y, width_x, width_y, rho, base):
        #    return lambda x,y: (twod_triangle(x, y, peak_x, peak_y, x0, y0,
        #                                     s1_x, s2_x, s1_y, s2_y) 
        #                       + gaussian(x, y, height, center_x, center_y,
        #                         width_x, width_y, rho) + base)
        #
        #param = (500, 100, 120, 348, 40, 20, 20, 20, 
        #         10, 120, 348, 20, 20, 10, 100)

        #errorfunction = lambda p: (np.ravel(err(*p)
        #                           (*np.indices(self.beam.shape)) 
        #                           - self.beam))
        #pfit, pcov, infodict, errmsg, success = leastsq(errorfunction,
        #                                                param, full_output=1,
        #                                                maxfev=500)

        #if success not in [1,2,3,4]:
        #    print('Center fitting failed!')
        #    print('Error message: {}'.format(errmsg))

        #xs, ys = np.indices(self.beam.shape)
        #f = err(*pfit)
        #fitted = f(*(xs, ys))
        #self.tpeak = np.max(fitted)
        #fig, axs = plt.subplots(2)
        #axs[0].imshow(self.beam)
        #axs[1].imshow(fitted)
        #plt.show()
        #self.center = np.where(fitted==self.tpeak)[0][0]
        pfit, s_sq = fit_with(gaussian_shift, self.beam, param_estimator=moments)
        fit = gaussian_shift(*pfit)
        xs, ys = np.indices(self.beam.shape)
        fitted = fit(*(xs, ys))
        self.tpeak = np.max(fitted)
        self.center = np.where(fitted==self.tpeak)[0][0]
        fig, axs = plt.subplots(3)
        axs[0].imshow(self.beam)
        sc = axs[1].imshow(fitted)
        plt.colorbar(sc)
        axs[2].plot(fitted[self.center])
        axs[2].plot(self.beam[self.center])
        print(self.dwell, self.power)
        t = f'{str(int(10**self.dwell))}us_{str(self.power)}W'
        plt.title(t)
        plt.savefig(t)
        plt.close()
        self.center = np.where(fitted==self.tpeak)[0][0]

    def fit_two_gaussian(self, plot=True):
        profile = self.beam[self.center, :]

        def oned_gaussian(x, height, x0, s):
            return height * np.exp(-((x-x0)/s)**2)

        def two_gaussian(height_0, x_0, s_0, height_1, x_1, s_1, base):
            return lambda x: (oned_gaussian(x, height_0, x_0, s_0)
                              - np.abs(oned_gaussian(x, height_1, x_1, s_1))
                              + base)
        m = np.max(np.abs(profile))
        print(m)
        errorfunction = lambda p: (np.ravel(two_gaussian(*p)
                                  (*np.indices(profile.shape)) - profile))
        param = np.array([m, 420, 400, 5, 525, 120, 25.0001])
                  #h_0,    x_0,    s_0,    h_1,    x_1,    s_1,    base
        bounds = ([0,      400,    100,      0,    500,    100,    25],
                  [2*m,    800,    600,      10,    550,    200,    25.001])

        result = least_squares(errorfunction, param,
                               bounds=bounds,
                               max_nfev=10000)
        print(result.x)
        xx = np.indices(profile.shape)
        if plot:
            plt.plot(xx.T, profile)
            plt.plot(xx.T, oned_gaussian(xx.T, *result.x[:3])+result.x[-1])
            plt.plot(xx.T, np.abs(oned_gaussian(xx.T, *result.x[3:-1])))
            plt.title(self.dwell + 'us ' + self.power + 'A') 
            plt.savefig('two gaussain '+ self.dwell + 'us ' + self.power + 'A') 
            plt.close() 
        save = [True, False, True, False, False, False, True]
        self.profile_params = result.x[save]
        self.tpeak = np.max(oned_gaussian(xx.T, *result.x[:3])+result.x[-1])
        self.uncertainty = result.jac.T @ result.jac

    def fit_profile(self):
        '''
        Fitting the profile along center
        '''
        profile = self.beam[self.center, :]
        
        #def _edgeworth(height, x0, s, sk, ku, base):
        #    return lambda x: height * edgeworth(x, x0, s, sk, ku) + base

        #param = (1000, 290, 1, 1, 1, 100)
        #errorfunction = lambda p: (np.ravel(_edgeworth(*p)
        #                          (*np.indices(profile.shape)) - profile))
        
        def oned_gaussian(x, x0, s):
            return np.exp(-((x-x0)/s)**2)

        def _gaussian(height, x0, s, base):
            return lambda x: height * oned_gaussian(x, x0, s) + base
        errorfunction = lambda p: (np.ravel(_gaussian(*p)
                                  (*np.indices(profile.shape)) - profile))
        param = (500, 290, 50, 50)
        pfit, pcov, infodict, errmsg, success = leastsq(errorfunction,
                                                        param, full_output=1,
                                                        maxfev=500)
         
        if success not in [1,2,3,4]:
            print('Fitting failed.')
            print('Error message: {}'.format(errmsg))
        
        #fit = _edgeworth(*pfit)
        fit = _gaussian(*pfit)
        xs = np.indices(profile.shape)
        # Uncertainty estimate
        s_sq = np.sum(errorfunction(pfit)**2)/(profile.shape[0] - len(param))
        pcov = pcov*s_sq
        self.uncertainty = pcov

        plt.plot(xs.T, profile)
        plt.plot(xs.T, fit(*xs))
        title = f'{int(10**self.dwell)}us_{int(self.power)}W_gauss'
        plt.title(f'{int(10**self.dwell)}us_{int(self.power)}W') 
        plt.savefig(title)
        plt.close()
        self.profile_params = pfit
        self.tpeak = np.max(fit(*xs))
