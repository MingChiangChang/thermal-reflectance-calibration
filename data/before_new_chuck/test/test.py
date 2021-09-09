import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--calibration', type=bool, 
            help="Run dark/blank/dark blank", default=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    print(args)
