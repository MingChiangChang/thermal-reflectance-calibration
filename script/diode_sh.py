''' Script for creating shell script for diode runs '''
import sys

import numpy as np

sys.path.insert(0, '../src')
from sh_script_gen import add_header, add_commend

conditions = {
            '377':
               { 'power': np.linspace(55, 110, 12),
                 'runs': 20},
           '567':
              { 'power': np.linspace(40, 110, 12),
                'runs': 20},
           '855':
              { 'power': np.linspace(40, 90, 12),
                'runs': 20},
           '1288':
              { 'power': np.linspace(30, 75, 12),
                'runs': 10},
           '1941':
              { 'power': np.linspace(30, 70, 12),
                'runs': 10},
            '2924':
               { 'power': np.linspace(25, 60, 12),
                 'runs': 10},
            '4405':
               { 'power': np.linspace(25, 58, 12),
                 'runs': 5},
            '6637':
               { 'power': np.linspace(25, 55, 12),
                 'runs': 5},
            '10000':
               { 'power': np.linspace(20, 53, 12),
                 'runs': 5}
            }

pre = 'Chess_diode'

with open('diode.sh', 'w') as f:
    counter=0
    for dw in conditions:
        runs = conditions[dw]['runs']
        if int(dw)<=1000:
            add_commend(f, n=runs, pmin='0 -40', pmax='0 40',
                    d=dw, p=0, pre=f'Calibration_{pre}', c='True',
                    r=100, a='LSA')
            for i in conditions[dw]['power']:
                add_commend(f, n=runs, p=round(i, 2), d=dw, m='r', yr='-40 40',
                            xr='1.5 2', pre=pre, r=100, a='LSA')
        else:
            add_commend(f, n=runs, pmin='{} -27'.format(-40+counter),
                        pmax='{} 27'.format(-40+counter), d=dw, p=0,
                        pre='Calibration_diode', c='True', r=100, a='LSA')
            for i in conditions[dw]['power']:
                add_commend(f, n=runs, d=dw, p=round(i, 2), m='r',
                            yr='-27 27',
                            xr='{:2f} {:2f}'.format(-40+counter,
                                                    -39.5+counter),
                            pre=pre, r=100, a='LSA')
                counter += 1
