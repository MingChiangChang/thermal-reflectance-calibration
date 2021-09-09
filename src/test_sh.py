from sh_script_gen import *

REP = 20
xr = ' '.join([str(-38+4*x) for x in range(REP)])
pre = 'Wafer_test'

with open('test.sh', 'w') as f:
    add_header(f)
    p = 63
    dw = 2500 
    add_commend(f, n=1, pmin='0 -27', pmax='0 27',
                    d=dw, p=0, pre='test_calibration', c='True',
                    r=100, a='LSA')
    add_commend(f, n=REP, m='r', yr='-27 27', p=p, d=dw,
               xr = xr, pre=pre, r=100, a='LSA')
