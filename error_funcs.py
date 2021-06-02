'''Error funcs for fitting in thermal reflectance calibration'''
import numpy as np

def linear(a,b):
    return lambda x: a*x+b

def twod_surface(base, a, b, c, d, e):
    return lambda x, y: base + a*x + b*y + c*x**2 + d*y**2 + e*x*y 

def jacobian_twod_surface(base, a, b, c, d, e):
    return lambda x, y: [1, x, y, x**2, y**2, x*y]

def power_fit_func(base, a, b, c, d, e):
    return lambda x, y: base + a*x + b*y + c*x**0.5 + d*y**2 + e*x*y

def triangle(x, peak, x0, s1, s2):
    tri =  peak-np.abs(x-x0)*(((x-x0)>0).astype(int)*s1+((x-x0)<0).astype(int)*s2)
    tri[tri<0]=0
    return tri

def twod_triangle(x, y, peak_x, peak_y, x0, y0, s1_x, s2_x, s1_y, s2_y):
    return triangle(x, peak_x, x0, s1_x, s2_x)\
           * triangle(y, peak_y, y0, s1_y, s2_y)

def elipse_cone(x, y, height, x0, y0, a, b):
    return height - ((x-x0)/a)**2 - ((y-y0)/b)**2

def gaussian(x, y, height, center_x, center_y, width_x, width_y, rho):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return height*np.exp( -(
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
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    #print(np.abs((np.arange(col.size)-x)**2*col))
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/np.abs(col.sum()))
    row = data[int(x), :]
    #print(np.abs((np.arange(row.size)-y)**2*row))
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/np.abs(row.sum()))
    height = data.max()
    rho = 0.0
    shift = 0.0
    print('Moments: {}, {}, {}, {}, {}'.format(height, x, y, width_x, width_y))
    return height, x, y, width_x, width_y, rho, shift

def edgeworth(x, x0, s, sk, ku):
    return 1/(2*np.pi*s)\
           * np.exp(-(x-x0)**2/(2*s**2))\
           * edge_expansion( (x-x0)/s, sk, ku)

def edge_expansion(r, k3, k4):
    return 1 + k3*(r**3-3*r)/6 + k4/12*(r**4-6*r**2+3)\
           + k3**2*(r**6-15*r**4+45*r**2-15)/72

def twod_edgeworth(height, x0, y0, s_x, s_y, sk_x, sk_y, ku_x, ku_y):
    return lambda x, y: height * edgeworth(x, x0, s_x, sk_x, ku_x)\
                        * edgeworth(y, y0, s_y, sk_y, ku_y)

def edgeworth_default_param_approx(data):
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    
    col = data[:, int(y)]
    s_x0 = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/np.abs(col.sum()))
    sk_x0 = ((np.arange(col.size)-x)**3*col).sum()/s_x0**3
    ku_x0 = ((np.arange(col.size)-x)**4*col).sum()/s_x0**4-3
    
    row = data[int(x), :]
    s_y0 = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/np.abs(row.sum()))
    sk_y0 = ((np.arange(row.size)-y)**3*row).sum()/s_y0**3
    ku_y0 = ((np.arange(row.size)-y)**4*row).sum()/s_y0**4-3
    height = data.max()
    print((f"Default height: {height}, x: {x}, y: {y}, s_x: {s_x0}, "
           f"s_y: {s_y0}, sk_x: {sk_x0}, sk_y: {sk_y0}, ku_x: {ku_x0}, "
           f"ku_y: {ku_y0}"))
    return height, x, y, s_x0, s_y0, sk_x0, sk_y0, ku_x0, ku_y0
