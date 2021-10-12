import sys

sys.path.insert(0, '../src')
from sh_script_gen import add_header, add_commend

dwells = ['567', '855', '1941', '2924', '4405', '6637', '10000']
start_pws = ['75', '75', '62', '59', '54', '53', '52']

pos = -35 
interval = 1 

with open('melt_calib_2.sh', 'w') as f:
    add_header(f)
    for dwell, start_pw in zip(dwells, start_pws):
        for i in range(5):
            if pos <= 35:
                add_commend(f, n=1, pmin='{:2f} 0'.format(pos),
                        pmax='{:2f} 30'.format(pos), d=dwell, 
                           p='{}'.format(int(start_pw)+i*1),
                           pre='melt_calib_2', r=200)
            else:
                add_commend(f, n=1, pmin='{:2f} -30'.format(pos-70),
                           pmax='{:2f} 0'.format(pos-70), d=dwell,
                           p='{}'.format(int(start_pw)+i*1),
                           pre='melt_calib_2', r=200)
            pos += interval 
            
