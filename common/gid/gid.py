from argparse import ArgumentParser
from google_images_download_patched import googleimagesdownload
import re

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--kwd', help=':specify keyword') # use action='store_true' as flag
    argparser.add_argument('--ctg', help=':specify category') # use action='store_true' as flag
    argparser.add_argument('--lim', help=':optional, specify limit') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.ctg: opts.update({'ctg':args.ctg})
    if args.kwd: opts.update({'kwd':args.kwd})
    if args.lim: opts.update({'lim':args.lim})

def main(limit='10'):
    outdir = opts['kwd']
    outdir = re.sub('\s', '_', outdir)
    imgdir = opts['ctg']
    imgdir = re.sub('\s', '_', imgdir)
    arguments = {
        'keywords':f"{opts['kwd']}",
        'output_directory':f"{outdir}",
        'prefix_keywords':f"{opts['ctg']}",
        'image_directory':f"{imgdir}",
        'limit':f"{limit}",
        'format':'jpg',
        'no_numbering':True,
        'print_urls':True,
    } 

    print(arguments)
    response = googleimagesdownload()
    response.download(arguments)

parseOptions()
if ('ctg' in opts.keys() and 'kwd' in opts.keys()):
    if ('lim' in opts.keys()):
        main(opts['lim'])
    else:
        main()
