from argparse import ArgumentParser

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--main', help=':main execution') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.main: opts.update({'main':args.main})

def main():
    infile = opts['main']
    with open(infile) as file:
        for line in file:
            print(line.rstrip())

if __name__ == '__main__':
	parseOptions()
	if ('main' in opts.keys()): main()
