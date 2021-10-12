import yaml
import pprint
import os
import matplotlib.pyplot as plt
import numpy as np 
import scipy.linalg as la
import sys

from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '../src')
from temp_calibration import estimate_noise, fitgaussian, gaussian_shift,\
                             fit_mask, trail, get_box, fit_edgeworth,\
                             twod_edgeworth, default_value_approx,\
                             read_and_add_weight_to_images

pp = pprint.PrettyPrinter(indent=4)
pprint = pp.pprint

kappa = 1.42*(10**(-4)) ; #Thermoreflectance coefficient - in this case calc'd by Steve
kappa = 1.2*(10**(-4))  ; #Thermoreflectance coefficient - literature
tmask = 900.

# MAC file path
directory = '/Users/mingchiang/Desktop/Work/thermal-reflectance-calibration/'
# Linux file path
#directory = '/home/ming-chiang/Desktop/Code/thermal-reflectance-calibration/'
with open(directory + 'data.yaml') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# Just checking
pprint(data)

for folder in data:
    print(folder)
    if folder.endswith('A'):
        for set_i in data[folder].keys():
            print(folder)
            print(set_i, os.getcwd())

            # Change directory
            os.chdir(directory+folder)
            if set_i not in data[folder]:
                continue
            img_dict = {}

            ######## Load and process images ###########
            dark_blnk = plt.imread(data[folder][set_i]['dark_blank'][0])
            for light_condition in data[folder][set_i]:
                img_dict[light_condition] = read_and_add_weight_to_images(data[folder][set_i]\
                                                                          [light_condition])
            for cond in img_dict:
                img_dict[cond] = np.array(img_dict[cond]).astype(float)
            
            print(img_dict['live'][0:200, :].dtype) 
            print('Noise', estimate_noise(img_dict['live']))
            print('Mean', np.mean(img_dict['live'][0:200, :] - img_dict['blank'][0:200, :]),
                    np.max(img_dict['live']), np.max(img_dict['blank']))
            
            live_sub = img_dict['live']-img_dict['blank']
            dark_sub = img_dict['dark']-img_dict['dark_blank']
            dI = live_sub - dark_sub
            therm = dI/img_dict['blank']
            therm_0 = np.mean(therm[0:200, :])
            temp = therm/kappa
            
            x_shift, x_width, y_shift, y_width = get_box(temp)
            print(x_shift, x_width, y_shift, y_width)
            print(np.max(temp))
            #x_shift = 400
            #x_width = 300
            #y_shift = 0
            #y_width = 900
            ############ Fitting Image ################
            m = np.zeros(temp.shape)
            m[x_shift:x_shift+x_width, y_shift:y_shift+y_width] = 1
            m = m.astype(int)
            profile_to_fit = temp[x_shift:x_shift+x_width, y_shift:y_shift+y_width]

            #_, axs = plt.subplots(2)
            #axs[0].imshow(temp)
            #axs[1].imshow(profile_to_fit)
            #plt.show()

            params, error = fitgaussian(profile_to_fit)
            (height, x, y, width_x, width_y, rho, shift) = params
            fit = gaussian_shift(*params)
            print(params)
            plt.matshow(profile_to_fit, cmap=plt.cm.copper)
            plt.contour(fit(*np.indices(profile_to_fit.shape)),
                        levels=[0.,100., 200., 300., 400., 500., 600., 700., 800., 900.,
                                1000., 1200., 1300., 1400.], cmap=plt.cm.cividis )
            ax = plt.gca()
            plt.show()

            mask_fit, dx = fit_mask(profile_to_fit, fit, tmask+100)
            print(x,y)
            mask_tr = trail(profile_to_fit, x, y, dx*2)
            mask = np.logical_and(mask_fit, mask_tr)
            
            #m = mask
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            #xs, ys = np.indices(profile_to_fit.shape)
            #print(xs.shape, profile_to_fit.shape)
            #xs = xs[m>0]
            #ys = ys[m>0]
            #xs = np.ravel(xs)
            #ys = np.ravel(ys)
            #print(ys.shape, profile_to_fit[m>0].shape)
            #ax.scatter(xs, ys, np.ravel(profile_to_fit[m>0]), s=3, c=np.ravel(profile_to_fit[m>0]))
            #ax.scatter(xs, ys, fit(*(xs, ys)))
            ##ax.view_init(0, 0)
            #plt.show()

            plt.matshow(profile_to_fit*mask.astype(int))
            plt.show()
            params, error = fit_edgeworth(profile_to_fit, mask)#, params)
            print(error)
            (height, x, y, width_x, width_y, sk_x, sk_y, ku_x, ku_y) = params
            x = x+x_shift
            y = y+y_shift
            params = (height, x, y, width_x, width_y, sk_x, sk_y, ku_x, ku_y)
            fit = twod_edgeworth(*params)

            #(height, x, y, width_x, width_y, rho, shift) = params
            #A = [[width_x**2, rho*width_x*width_y],[rho*width_x*width_y, width_y**2]]
            #print(A)
            #eigvals, eigvecs = la.eig(A)
            #eigvals = eigvals.real
            #print("eigval", np.sqrt(eigvals[0]), np.sqrt(eigvals[1]))
            
            ############## Plotting ####################
            plt.imshow(temp>np.max(temp)*0.6)
            plt.show()
            plt.matshow(temp)#, cmap=plt.cm.copper)
            plt.contour(fit(*np.indices(temp.shape)),
                        levels=[0.,100., 200., 300., 400., 500., 600., 700., 800., 900.,
                                1000., 1200., 1300., 1400.], cmap=plt.cm.cividis )
            ax = plt.gca()

            #plt.text(0.95, 0.05, """
            #amp : %.1f
            #x : %.1f
            #y : %.1f
            #width_x : %.1f
            #width_y : %.1f
            #eig_x : %.1f
            #eig_y : %.1f
            #shift : %.1f
            #rho : %f
            #tmax : %f""" %(height, x, y, width_x, width_y, np.sqrt(eigvals[0]),
            #    np.sqrt(eigvals[1]), shift, rho, height + shift),
            #        fontsize=16, color='white', horizontalalignment='right',
            #        verticalalignment='bottom', transform=ax.transAxes)
            #plt.show()
            
            sc = plt.matshow(temp-fit(*np.indices(temp.shape)))
            plt.colorbar(sc)
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xs, ys = np.indices(temp.shape)
            print(xs.shape, temp.shape)
            xs = xs[m>0]
            ys = ys[m>0]
            #xs = np.ravel(xs)
            #ys = np.ravel(ys)
            print(ys.shape, temp[m>0].shape)
            #ax.scatter(xs, ys, np.ravel(temp[m>0]))
            ax.plot_surface(xs, ys, fit(*(xs, ys)))
            ax.view_init(0, 0)
            plt.show()
