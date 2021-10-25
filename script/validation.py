from pathlib import Path
import sys
import json
import argparse

from scipy.optimize import fsolve
import numpy as np
import dill

sys.path.insert(0, '../src')

from error_funcs import temp_surface

def parse():
    parser = argparse.ArgumentParser(description='Validate temperature surface')
    parser.add_argument('-t', '--tpeak', type=float, 
                        help='Tpeak value in C')
    parser.add_argument('-p', '--power', type=float,  
                        help='Power value in A')
    parser.add_argument('-d', '--dwell', type=float,
                        help='Dwell value in us')
    args = parser.parse_args()
    return args

def validate(args):
    if args.dwell is not None:
        if args.tpeak is not None:
            return p_func(args.tpeak, np.log10(args.dwell))
        elif args.power is not None:
            return t_func(np.log10(args.dwell), args.power)
    else:
        if args.tpeak is not None and args.power is not None:
            return fsolve(lambda x: (t_func(np.log10(x), args.power)
                                     - args.tpeak), [3])

if __name__ == "__main__":
    
    home = Path.home()
    path = home / 'Desktop' / 'github' / 'thermal-reflectance-calibration'
    path = path / 'data' 

    with open(path / "t_func.d", 'rb') as f:
        t_func = dill.load(f) 
    with open(path / "p_func.d", 'rb') as f:
        p_func = dill.load(f)

    args = parse()
    print(validate(args))
