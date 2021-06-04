import numpy as np
import matplotlib.pyplot as plt

from temp_calibration import fit_xy_to_z_surface_with_func
from error_funcs import twod_surface

g = np.random.randn(100,100)
x, y = np.indices(g.shape) 
p = [1,2,3,4,5,6]
z = p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*y**2 + p[5]*x*y

param = [0,0,0,0,0,0]
print(fit_xy_to_z_surface_with_func(x, y, z, twod_surface, param) )
