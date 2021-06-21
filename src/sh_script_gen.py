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
    f.write('python3 ThermalReflectance.py ')
    for key, item in kwargs.items():
        f.write(f'-{key} {item} ')
    f.write('\n')

def add_condition_grid(f, dw_ls, pw_ls, xr, ymin, ymax, **kwargs):
    xs = np.min(xr)
    for dw in dw_ls:
        for pw in pw_ls:
            new_dict = dict(kwargs, d=dw, p=pw, 
                            pmin=f'{xs} {ymin}',
                            pmax=f'{xs} {ymax}')
            add_commend(f, **new_dict)
            #xs += 0.01
            if xs > np.max(xr): 
                print('Failed. Position requested out of bound.')

def add_dws_n_powers(f, dws, pws, **kwargs):
    for dw, pw in zip(dws, pws):
        new_dict = dict(kwargs, d=dw, p=pw)
        add_commend(f, **new_dict)

def add_moving_scan(f, interval, **kwargs):
    num = kwargs['n']
    for i in range(num):
        kwargs['n'] = 1
        x, y = parse_position(kwargs['pmin'])
        new_x = str(float(x)+interval)
        kwargs['pmin'] = f'{new_x} {y}'
        add_commend(f, **kwargs)

def parse_position(s):
    xy = s.split(' ')
    x = xy[0]
    y = xy[1]
    return x, y

if __name__ == '__main__':
    conditions = {
#            '250': 
#               { 'power': np.linspace(20, 80, 13),
#                 'runs': 30 },
#            '377': 
#               { 'power': np.linspace(20, 80, 13),
#                 'runs': 30 },
            '567': 
               { 'power': np.linspace(20, 65, 10),
                 'runs': 20 },
            '855': 
               { 'power': np.linspace(20, 60, 9),
                 'runs': 20 },
            '1288': 
               { 'power': np.linspace(20, 55, 8),
                 'runs': 10},
            '1941': 
               { 'power': np.linspace(20, 50, 7),
                 'runs': 10},
#            '2924': 
#               { 'power': np.linspace(20, 50, 7),
#                 'runs': 5},
#            '4405': 
#               { 'power': np.linspace(20, 45, 6),
#                 'runs': 5},
#            '6637': 
#               { 'power': np.linspace(20, 40, 5),
#                 'runs': 5},
#            '10000': 
#               { 'power': np.linspace(20, 40, 5),
#                 'runs': 5}
            }

    with open('0618.sh', 'w') as f:
        add_header(f)
        counter = 0
        for dw in conditions:
            runs = conditions[dw]['runs']
            if int(dw)<=1941:
                add_commend(f, n=runs, pmin='1 -40', pmax='1 40',
                           d=dw, p=0, pre='Calibration_0618', c='True', r=100)
                for i in conditions[dw]['power']:
                    add_commend(f, n=runs, d=dw, p=i, m='r', yr='-40 40', xr='1.5 2', pre='0618', r=100)
            else:
                add_commend(f, n=runs, pmin='{} -25'.format(-40+counter*0.8),
                             pamx='{} 25'.format(-40+counter*0.8),
                            d=dw, p=0, pre='Calibration_0617', c='True', r=100)
                for i in conditions[dw]['power']:
                    add_commend(f, n=runs, d=dw, p=i, m='r',
                            yr='-25 25',
                            xr='{:2f} {:2f}'.format(-40+counter*0.8, -39.6+counter*0.8),
                            pre='0617', r=100)
                    counter += 1
        #add_condition_grid(f, [250, 500, 1000],
        #                  [15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        #                  n=5, xr=(0, 20), ymin=-40, ymax=40, pre='0603')
        #counter = 0
        #for dw in conditions:
        #    for t in conditions[dw]['power']:
        #        runs = conditions[dw]['runs']
        #        add_commend(f, n=runs, d=dw, p=t, m='r',
        #                yr='-25 25',
        #                xr='{:2f} {:2f}'.format(-40+counter*0.4, counter*0.4),
        #                pre='0608')
        #        counter += 1
