def parse(fn):
    l = fn.split('_')
    for i in l:
        if 'us' in i:
            dwell = int(i[:i.index('us')])
        if 'W' in i:
            power = int(i[:i.index('W')])
    return dwell, power
