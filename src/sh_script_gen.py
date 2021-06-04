import numpy as np

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


def add_header(f):
    f.write('#!/bin/sh')
    f.write('\n')
    f.write('cd /Users/mingchiang/Desktop/Work/sara-socket-client/Scripts/')
    f.write('\n')

def add_commend(f, **kwargs):
    f.write('python ThermalReflectance.py ')
    for key, item in kwargs.items():
        f.write(f'-{key} {item} ')
    f.write('\n')


def add_condition_grid(f, dw_ls, pw_ls, xr, ymin, ymax, **kwargs):
    xs = np.min(xr)
    for dw in dw_ls:
        for pw in pw_ls:
            new_dict = dict(kwargs, d=dw, p=pw, pmin=f'{xs} {ymin}', pmax=f'{xs} {ymax}')
            add_commend(f, **new_dict)
            #xs += 0.01
            if xs > np.max(xr): 
                return

if __name__ == '__main__':
    with open('test.sh', 'w') as f:
        add_header(f)
        add_commend(f, n=1, pmin='0 -20', pmax='0 -20',
                   d=800, p=30, pre='TEST', c='True')
        add_condition_grid(f, [250, 500, 1000], [15, 20, 25, 30, 35, 40, 45, 50, 55, 60], n=5,
                           xr=(0, 20), ymin=-40, ymax=40, pre='0603')

