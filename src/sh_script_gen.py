'''
Acceptible keywords:
    -pt, --plot      Plot on screen
 -pre PREFIX, --prefix PREFIX
            Prefix directory for storing data
 -a {Analysis,LSA,Local}, --address {Analysis,LSA,Local}
            Address of the device
 -r RINGSIZE, --ringsize RINGSIZE
            Ring size
 -p POWER, --power POWER
            Anneal power
 -d DWELL, --dwell DWELL
            Dwell time in mus
 -pmin Xmin Ymin, --posmin Xmin Ymin
            X,Y position of the scan start
 -pmax Xmax Ymax, --posmax Xmax Ymax
            X,Y position of the scan end
 -fmin FRAMEMIN, --framemin FRAMEMIN
            Lower bound of frame
 -fmax FRAMEMAX, --framemax FRAMEMAX
            Upper bound of frame (set to -1 to print all)
 -n NLOOPS, --nloops NLOOPS
            Number of loopshj
'''

import numpy as np

# pylint: disable=C0116
# Mostly self explanatory

def add_header(output_file):
    output_file.write('#!/bin/sh')
    output_file.write('\n')
    # f.write('cd /Users/mingchiang/Desktop/Work/sara-socket-client/Scripts/')
    output_file.write('\n')

def add_commend(output_file, **kwargs):
    output_file.write('python3 ThermalReflectance.py ')
    for key, item in kwargs.items():
        output_file.write(f'-{key} {item} ')
    output_file.write('\n')

def add_condition_grid(output_file, dw_ls, pw_ls, x_r, ymin, ymax, **kwargs):
    x_s = np.min(x_r)
    for dwell in dw_ls:
        for power in pw_ls:
            new_dict = dict(kwargs, d=dwell, p=power,
                            pmin=f'{x_s} {ymin}',
                            pmax=f'{x_s} {ymax}')
            add_commend(output_file, **new_dict)
            #xs += 0.01
            if x_s > np.max(x_r):
                print('Failed. Position requested out of bound.')

def add_dws_n_powers(output_file, dws, pws, **kwargs):
    for dwell, power in zip(dws, pws):
        new_dict = dict(kwargs, d=dwell, p=power)
        add_commend(output_file, **new_dict)

def add_moving_scan(output_file, interval, **kwargs):
    num = kwargs['n']
    for _ in range(num):
        kwargs['n'] = 1
        x, y = parse_position(kwargs['pmin'])
        new_x = str(float(x)+interval)
        kwargs['pmin'] = f'{new_x} {y}'
        add_commend(output_file, **kwargs)

def parse_position(s):
    x_and_y = s.split(' ')
    x = x_and_y[0]
    y = x_and_y[1]
    return x, y

if __name__ == '__main__':
    conditions = {
            '2924':
               { 'power': np.linspace(20, 50, 7),
                 'runs': 7},
            '4405':
               { 'power': np.linspace(20, 45, 6),
                 'runs': 6},
            '6637':
               { 'power': np.linspace(20, 40, 5),
                 'runs': 5},
            '10000':
               { 'power': np.linspace(20, 40, 5),
                 'runs': 4}
            }

    with open('0622.sh', 'w') as f:
        add_header(f)
        counter = 0 # pylint: disable=invalid-name
        for dw in conditions:
            runs = conditions[dw]['runs']
            if int(dw)<=1941:
                add_commend(f, n=runs, pmin='1 -40', pmax='1 40',
                           d=dw, p=0, pre='Calibration_0622', c='True', r=100)
                for i in conditions[dw]['power']:
                    add_commend(f, n=runs, d=dw, p=i, m='r', yr='-40 40',
                                xr='1.5 2', pre='0622', r=100)
            else:
                add_commend(f, n=runs, pmin='{} -25'.format(0+counter*0.8),
                             pmax='{} 25'.format(0+counter*0.8),
                            d=dw, p=0, pre='Calibration_0622', c='True', r=200)
                for i in conditions[dw]['power']:
                    add_commend(f, n=runs, d=dw, p=i, m='r',
                            yr='-25 25',
                            xr='{:2f} {:2f}'.format(0+counter*0.8, 0.4+counter*0.8),
                            pre='0622', r=200)
                    counter += 1
