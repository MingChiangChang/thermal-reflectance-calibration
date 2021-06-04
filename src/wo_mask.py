import yaml
import pprint
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from mpl_toolkits.mplot3d import Axes3D

from temp_calibration import estimate_noise, fit_with, fit_mask, trail,\
                             get_box, read_and_add_weight_to_images
from error_funcs import gaussian_shift, moments, twod_edgeworth,\
                        edgeworth_default_param_approx

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
            np.save('temp', temp)
            x_shift, x_width, y_shift, y_width = get_box(temp)
            print('box param: ', x_shift, x_width, y_shift, y_width)

            m = np.zeros(temp.shape)
            m[x_shift:x_shift+x_width, y_shift:y_shift+y_width] = 1
            m = m.astype(int)
            profile_to_fit = temp[x_shift:x_shift+x_width, y_shift:y_shift+y_width]
            np.save('profile', profile_to_fit)
            #_, axs = plt.subplots(2)
            #axs[0].imshow(temp)
            #axs[1].imshow(profile_to_fit)
            #plt.show()

            params, error = fit_with(gaussian_shift, moments, profile_to_fit)
            (height, x, y, width_x, width_y, rho, shift) = params
            fit = gaussian_shift(*params)
            print(params)

            plt.matshow(profile_to_fit, cmap=plt.cm.copper)
            plt.contour(fit(*np.indices(profile_to_fit.shape)),
                        levels=[100*x for x in range(15)], cmap=plt.cm.cividis)
            ax = plt.gca()
            plt.show()
            
            params, error = fit_with(twod_edgeworth, 
                                     edgeworth_default_param_approx,
                                     profile_to_fit)
            (height, x, y, width_x, width_y, sk_x, sk_y, ku_x, ku_y) = params
            x = x+x_shift
            y = y+y_shift
            params = (height, x, y, width_x, width_y, sk_x, sk_y, ku_x, ku_y)
            fit = twod_edgeworth(*params)

            plt.matshow(temp)#, cmap=plt.cm.copper)
            plt.contour(fit(*np.indices(temp.shape)),
                        levels=[0.,100., 200., 300., 400., 500., 600., 700., 800., 900.,
                                1000., 1200., 1300., 1400.], cmap=plt.cm.cividis )
            ax = plt.gca() 
            plt.show()

            sc = plt.imshow(temp-fit(*np.indices(temp.shape)), vmin=0, vmax=500)
            plt.colorbar(sc)
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            xs, ys = np.indices(temp.shape)
            print(xs.shape, temp.shape)
            xs = xs[m>0].reshape(x_width, y_width)
            ys = ys[m>0].reshape(x_width, y_width)
            temp = temp[m>0].reshape(x_width, y_width)
            #xs = np.ravel(xs)
            #ys = np.ravel(ys)
            #ax.scatter(xs, ys, np.ravel(temp[m>0]))
            Z = np.array(fit(*(xs, ys))).reshape(xs.shape)
            print(Z.shape)
            ax.scatter(xs, ys, temp, s=3)
            ax.plot_surface(xs, ys, np.array(fit(*(xs, ys))).reshape(xs.shape))
            ax.view_init(0, 0)
            plt.show()
