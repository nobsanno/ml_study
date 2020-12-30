from argparse import ArgumentParser
import modeling_icfs_com as mic

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--wmn', help=':specify write model name') # use action='store_true' as flag
    argparser.add_argument('--ncl', help=':specify number of class, default value is 2') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.wmn: opts.update({'wmn':args.wmn})
    if args.ncl: opts.update({'ncl':args.ncl})

image_size = (180, 180)

if __name__ == '__main__':
    parseOptions()
    if ('wmn' in opts.keys()):
        num_classes = 2
        if ('ncl' in opts.keys()):
            num_classes = int(opts['ncl'])
        model = mic.make_model(input_shape=image_size + (3,), num_classes=num_classes)
        model.summary()
        model.save(opts['wmn'])
