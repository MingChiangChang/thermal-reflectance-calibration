''' Utility functions '''

def parse(file_name):
    '''
    Function for parsing filenames and extracting dwell and power
    '''
    conds = file_name.split('_')
    for i in conds:
        if 'us' in i:
            dwell = int(i[:i.index('us')])
        if 'W' in i:
            power = int(i[:i.index('W')])
    return dwell, power
