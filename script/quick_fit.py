import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

def Tpeak(power, tau):
    return -1.138*10**2* power - 2.45*10**3 * tau + 1.228*10**-1 * power**2 + 1.485*10**2 * tau**2 +3.369*10 * power*tau + 6.588333*10**3

def linear(a, b):
    return lambda x: a*x + b

dwell = np.array([1000, 1292, 1668, 2154, 2738, 3594, 4642, 5996, 7743, 10000])
power = np.array([150, 135, 122, 111.5, 103, 95, 88.5, 83.5, 79, 75])
real_power = np.array([110, 100, 88, 82, 77, 72, 69, 66, 62, 61])
fix = power-real_power

log_dwell = np.log10(dwell)

err = lambda p: linear(*p)(log_dwell)  - fix

print(leastsq(err, x0=np.array([1, 1])))
