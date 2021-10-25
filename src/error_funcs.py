'''Error funcs for fitting in thermal reflectance calibration'''
import math

import numpy as np
import sympy as sp

def linear(a,b):
    ''' Linear function ax+b'''
    return lambda x: a*x+b

def twod_surface(base, a, b, c, d, e):
    ''' 2d quadratic surface '''
    return lambda x, y: base + a*x + b*y + c*x**2 + d*y**2 + e*x*y

def cubic_surface(base, a, b, c, d, e, f, g):
    ''' 2d cubic surface without cross terms'''
    return lambda x, y: base + a*x + b*y + c*x**2 + d*y**2 + e*x*y + f*x**4 + g*y**4

def temp_surface(base, a, b, c, d, e, f, g):
    ''' Temperture surface'''
    return lambda x, y: ( base + a*x + b*y + c*x**2 + d*y**2 
                          + e*x*y + f*x*y**2 + g*x**2* y )

def temp_surface_sp(base, a, b, c, d, e, f, g):
    ''' Temperture surface'''
    return lambda x, y: base + a*x + b*y + c*x**2 + d*y**2 + e*x*y + f*sp.sqrt(x) + g*sp.sqrt(y)

def jacobian_twod_surface():
    ''' Jacobian of a quadratic function at position (x, y)'''
    return lambda x, y: [1, x, y, x**2, y**2, x*y]

def power_fit_func(base, a, b, c, d, e):
    ''' Function for fitting inverted quadratic surface.
        a*x + b*y + c*sqrt(x) + d*y^2 + e*x*y '''
    return lambda x, y: base + a*x + b*y + c*x**0.5 + d*y**2 + e*x*y

def triangle(x, peak, x_0, s_1, s_2):
    ''' Triangular function '''
    tri =  peak-np.abs(x-x_0)*(((x-x_0)>0).astype(int)*s_1+((x-x_0)<0).astype(int)*s_2)
    tri[tri<0]=0
    return tri

def twod_triangle(x, y, peak_x, peak_y, x_0, y_0, s1_x, s2_x, s1_y, s2_y):
    ''' Function for 2d triangle '''
    return triangle(x, peak_x, x_0, s1_x, s2_x)\
           * triangle(y, peak_y, y_0, s1_y, s2_y)

def elipse_cone(x, y, height, x_0, y_0, a, b):
    ''' Function for elipscial cone '''
    return height - ((x-x_0)/a)**2 - ((y-y_0)/b)**2

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

def g_gaussian(height, center_x, center_y, width_x, width_y, e_g):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp( -(
        (np.abs(center_x-x)/width_x)**e_g +
        (np.abs(center_y-y)/width_y)**2
        )/np.sqrt(2.*e_g))

def edgeworth(x, x_0, s, sk, ku):
    ''' Edgeworth function '''
    return (1/(2*np.pi*s)
            * np.exp(- (x - x_0)**2/(2*s**2))
            * edge_expansion( (x - x_0)/s, sk, ku))

def edge_expansion(r, k_3, k_4):
    ''' Expansion part of edgeworth function '''
    return (1 + k_3*(r**3 - 3*r)/6
            + k_4/12*(r**4 - 6*r**2+3)
            + k_3**2*(r**6 - 15*r**4 + 45*r**2 - 15)/72)

def twod_edgeworth(height, x_0, y_0, s_x, s_y, sk_x, sk_y, ku_x, ku_y):
    ''' 2d Edgeworth function '''
    return lambda x, y: height * edgeworth(x, x_0, s_x, sk_x, ku_x)\
                        * edgeworth(y, y_0, s_y, sk_y, ku_y)
