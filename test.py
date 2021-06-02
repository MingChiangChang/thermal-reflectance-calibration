import numpy as np

def LaserPowerMing(tau, Temp):
    T = Temp
    P = (5000*np.sqrt(5)*np.sqrt(614000*T+1327591125*tau**2 -8080505000*tau+12142813538)-341554591*tau+1065621710)/3070000
    power = P 
    
    Temp_reverse = -1.138*10**2* (power-25.9187*tau+116.247) - 2.45*10**3 * tau + 1.228*10**-1 * (power-25.9187*tau+116.247)**2 + 1.485*10**2 * tau**2 +3.369*10 * (power-25.9187*tau+116.247)*tau + 6.588333*10**3
    print("Calling Ming's laser power: Tin",T, "tau", tau, "P", P, "T(P,tau)",Temp_reverse)
    return P
