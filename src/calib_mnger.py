''' Manager class for temperature profiling '''

from tqdm import tqdm
import numpy as np

from block import Block
from temp_calibration import fit_xy_to_z_surface_with_func
from error_funcs import jacobian_twod_surface

class CalibMnger():
    '''
    Main purpose of this class:
        1. Manage and maintain list of Block object
        2. Send request to each Block to fit
        3. Interfacing with user to quert data from each block
        4. Some convenient fitting function for temperature profiling
    '''

    def __init__(self, img_ls, dw_ls, pw_ls):
        '''
        Take image path list, dwell list, power list and
        create list of block objects
        '''
        self.block_lst = []
        self.power_lst = []
        self.dwell_lst = []  # log10 dwells
        self.tpeak_lst = []

        self.param_fitting_param = {}
        self.temp_surface_params = []
        self.cov = []

        self.dwell_lst = np.log10(np.array(dw_ls))
        self.power_lst = np.array(pw_ls)
        self.temp_cov = 0
        print(f'Reading {len(dw_ls)} images...')

        for img, dwell, power in tqdm(zip(img_ls, self.dwell_lst, self.power_lst)):
            image = np.load(img)
            self.block_lst.append(Block(image, dwell, power))

    def __len__(self):
        return len(self.block_lst)

    def fit_tpeak(self):
        ''' Make every block fit their data '''
        for block in tqdm(self.block_lst, desc='Fitting tpeak and profile'):
            # block.process_image()
            # block.get_beam()
            block.fit_center()
            block.fit_profile()
            self.tpeak_lst.append(block.tpeak)
            print(block)

    def get_tpeak_fitting_params(self, fitting_func, param, mask=None):
        ''' fitting tpeak to the fitting function'''
        if mask is None:
            mask = np.ones(self.power_lst.shape).astype(bool)#np.array(self.tpeak_lst) > 0
        power_data = np.array(self.power_lst)[mask]
        dwell_data = np.array(self.dwell_lst)[mask]
        tpeak_data = np.array(self.tpeak_lst)[mask]
        uncertainty = np.sqrt(np.array([block.uncertainty for block in self.block_lst]))
        p_fit, pcov, infodict = fit_xy_to_z_surface_with_func(power_data, dwell_data,
                                          tpeak_data, fitting_func,
                                          param, uncertainty = uncertainty[:,0,0])
        self.temp_surface_params = p_fit
        s_sq = (infodict['fvec']**2).sum()/(len(self.block_lst)-6) # What is this six??
        self.temp_cov = pcov*s_sq
        return p_fit, pcov, infodict

    def get_power_fitting_params(self, fitting_func, param, mask=None):
        ''' fitting tpeak to the fitting function'''
        if mask is None:
            mask = np.array(self.tpeak_lst) > 20
        power_data = np.array(self.power_lst)[mask]
        dwell_data = np.array(self.dwell_lst)[mask]
        tpeak_data = np.array(self.tpeak_lst)[mask]

        p_fit, pcov, _ = fit_xy_to_z_surface_with_func(tpeak_data,
                                          dwell_data, power_data,
                                          fitting_func, param)
        return p_fit, pcov

    def fitting_profile_params(self, fitting_func, param, mask=None):
        '''
        fit the fitting profile parameters with the fitting function
        Has the option of applying mask. (Defualt to only fit the ones with
                                          fitted tpeak>20C)
        '''
        if mask is None:
            mask = np.array(self.tpeak_lst) > 20 # make this higher?
        params_arr_to_fit = self.collect_fitting_params()

        power_data = np.array(self.power_lst)[mask]
        dwell_data = np.array(self.dwell_lst)[mask]

        for params in params_arr_to_fit:
            p_fit, _, _ = fit_xy_to_z_surface_with_func(power_data,
                         dwell_data, params_arr_to_fit[params][mask],
                         fitting_func, param)
            print(params, ': ', p_fit)
            self.param_fitting_param[params] = p_fit

    def collect_fitting_params(self):
        '''
        Collect Gaussian fitting parameters.
        return height, std, base
         '''
        height_lst = []
        s_lst = []
        base_lst = []
        for block in tqdm(self.block_lst,
                          desc="Collecting fitting params"):
            height_lst.append(block.profile_params[0])
            s_lst.append(block.profile_params[1])
            base_lst.append(block.profile_params[2])

        param_dict = {'Height': np.array(height_lst),
                      'Std': np.array(s_lst),
                      'Base': np.array(base_lst)}
        return param_dict

    def get_data_along_dw(self, dwell):
        ''' Collect power, tpeak and uncertainty data along certain dwell time '''
        tpeak = []
        power = []
        uncer = []
        for block in self.block_lst:
            print(block.dwell, dwell)
            if block.dwell == dwell:
                tpeak.append(block.tpeak)
                power.append(float(block.power))
                uncer.append(block.uncertainty[0,0])
        return power, tpeak, uncer

    def temp_surface_jacobian(self):
        ''' Return jacobian of the two d surface '''
        return jacobian_twod_surface(*self.temp_surface_params)

    def uncertainty_at(self, powers, dwells):
        ''' Calculate the uncertainty at a list of powers and dwells '''
        jacob = self.temp_surface_jacobian()
        j = []
        for power, dwell in zip(powers, dwells):
            j.append(jacob(power, dwell))
        j = np.array(j)
        print(j)
        print(self.temp_cov)
        return j @ self.temp_cov @ j.T

    def store_dw_pw_temp_at(self, path):
        ''' Function to store dwells, powers and fitted temperature as npy to a destination path '''
        dw_ls = []
        pw_ls = []
        temp_ls = []
        for block in self.block_lst:
            dw_ls.append(block.dwell)
            pw_ls.append(block.power)
            temp_ls.append(block.tpeak)
        data = np.array([dw_ls, pw_ls, temp_ls])
        np.save(path, data)
