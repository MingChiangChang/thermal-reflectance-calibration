from scipy.optimize import leastsq, least_squares

import numpy as np
import matplotlib.pyplot as plt

from temp_calibration import fit_with, gaussian_shift, moments

class Block():

    kappa = 1.2*10**-4

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

        self.uncertainty = 0

    def __repr__(self):
        x_0, x_1 = self.x_shift, self.x_shift+self.x_width
        y_0, y_1 = self.y_shift, self.y_shift+self.y_width
        return (f'Condition: {10**self.dwell}us, {self.power}W\n'
               + f'Fitted results: {self.tpeak}C \n'
               + f'Center: {self.center}\n'
               + f'Fitting paramters: {self.profile_params}\n'
               + f'Box info: {x_0}, {y_0}, {x_1}, {y_1}\n')

    def __str__(self):
        return self.__repr__()

    def get_beam(self):
        self.x_shift, self.x_width,\
        self.y_shift, self.y_width = (200, 400, 200, 900)
        self.beam = self.temp[self.x_shift:self.x_shift+self.x_width,
                              self.y_shift:self.y_shift+self.y_width]

    def fit_center(self):
        '''
        Fitting the center center of the beam using a 2d triangle +
        gaussian function
        '''
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
        errorfunction = lambda p: (np.ravel(two_gaussian(*p)
                                  (*np.indices(profile.shape)) - profile))
        param = np.array([m, 420, 400, 5, 525, 120, 25.0001])
                  #h_0,    x_0,    s_0,    h_1,    x_1,    s_1,    base
        bounds = ([0,      400,    100,      0,    500,    100,    25.000],
                  [2*m,    800,    600,     10,    550,    200,    25.001])

        result = least_squares(errorfunction, param,
                               bounds=bounds,
                               max_nfev=10000)
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

        def oned_gaussian(x, x0, s):
            return np.exp(-((x-x0)/s)**2)

        def _gaussian(height, x0, s, base):
            return lambda x: height * oned_gaussian(x, x0, s) + base
        
        errorfunction = lambda p: (np.ravel(_gaussian(*p)
                                  (*np.indices(profile.shape)) - profile))
        param = (500, 290, 50, 50)
        pfit, pcov, _, errmsg, success = leastsq(errorfunction,
                                                 param, full_output=1,
                                                 maxfev=500)

        if success not in [1,2,3,4]:
            print(f'Fitting failed for fitting {self.dwell}us {self.power}A.')
            print('Error message: {}'.format(errmsg))
            return
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
