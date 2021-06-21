from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from Block import Block
from temp_calibration import fit_xy_to_z_surface_with_func
from error_funcs import jacobian_twod_surface

class CalibMnger():

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
        print(f'Reading {len(dw_ls)} images...')
        
        for img, dw, pw in tqdm(zip(img_ls, self.dwell_lst, self.power_lst)):
            im = np.load(img)
            self.block_lst.append(Block(im, dw, pw))

    def __len__(self):
        return len(self.block_lst)

    #def parse_condition(self, condition):
    #    ''' Parse the file names to get condition'''
    #    sep = condition.index('_')
    #    return condition[:sep-2], condition[sep+1:-1]

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
            mask = np.array(self.tpeak_lst) > 0 
        power_data = np.array(self.power_lst)[mask]
        dwell_data = np.array(self.dwell_lst)[mask]
        tpeak_data = np.array(self.tpeak_lst)[mask]
        uncertainty = np.sqrt(np.array([block.uncertainty for block in self.block_lst]))
        p, pcov, infodict = fit_xy_to_z_surface_with_func(power_data, dwell_data,
                                          tpeak_data, fitting_func,
                                          param, uncertainty = uncertainty[:,0,0])
        self.temp_surface_params = p
        s_sq = (infodict['fvec']**2).sum()/(len(self.block_lst)-6) # What is this six??
        self.temp_cov = pcov*s_sq
        return p, pcov, infodict

    def get_power_fitting_params(self, fitting_func, param, mask=None):
        ''' fitting tpeak to the fitting function'''
        if mask is None:
            mask = np.array(self.tpeak_lst) > 20
        power_data = np.array(self.power_lst)[mask]
        dwell_data = np.array(self.dwell_lst)[mask]
        tpeak_data = np.array(self.tpeak_lst)[mask]

        p, pcov, infodict= fit_xy_to_z_surface_with_func(tpeak_data, dwell_data,
                                          power_data, fitting_func,
                                          param)
        return p, pcov

    def fitting_profile_params(self, fitting_func, param, mask=None):
        if mask is None:
            mask = np.array(self.tpeak_lst) > 20 
        params_arr_to_fit = self.collect_fitting_params() 
        
        power_data = np.array(self.power_lst)[mask]
        dwell_data = np.array(self.dwell_lst)[mask]

        for params in params_arr_to_fit:
            p, pcov, _ = fit_xy_to_z_surface_with_func(power_data, dwell_data,
                                              params_arr_to_fit[params][mask],
                                              fitting_func, param)
            print(params, ': ', p)
            self.param_fitting_param[params] = p

    def collect_fitting_params(self):
        ''' Collect fitting parameters. Currently fixed for gaussian'''
        height_lst = []
        s_lst = []
        base_lst = []
        for block in tqdm(self.block_lst, desc="Collecting fitting params"):
             height_lst.append(block.profile_params[0])
             s_lst.append(block.profile_params[1])
             base_lst.append(block.profile_params[2])
        
        param_dict = {'Height': np.array(height_lst),
                      'Std': np.array(s_lst),
                      'Base': np.array(base_lst)}
        return param_dict

    #def get_data_along_t(self):
    def get_data_along_dw(self, dwell):
        tpeak = []
        power = []
        uncer = []
        for block in self.block_lst:
            if int(block.dwell) == dwell:
                tpeak.append(block.tpeak)
                power.append(float(block.power))
                uncer.append(block.uncertainty[0,0])
        return power, tpeak, uncer

    def temp_surface_jacobian(self):
        return jacobian_twod_surface(*self.temp_surface_params)

    def uncertainty_at(self, ps, dws):
        jacob = self.temp_surface_jacobian()
        j = []
        for p, dw in zip(ps, dws):
            j.append(jacob(p, dw))
        j = np.array(j)
        print(j)
        print(self.temp_cov)
        return j @ self.temp_cov @ j.T
