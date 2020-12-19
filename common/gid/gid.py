from argparse import ArgumentParser
from google_images_download_patched import googleimagesdownload
import re

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--ctg', help=':specify category') # use action='store_true' as flag
    argparser.add_argument('--kwd', help=':specify keyword') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.ctg: opts.update({'ctg':args.ctg})
    if args.kwd: opts.update({'kwd':args.kwd})

def main(limit='100'):
    dir = opts['kwd']
    dir = re.sub('\s', '_', dir)
    arguments = {
        "keywords":f"{opts['ctg']} {opts['kwd']}",
        "output_directory":f"{dir}",
        "format":"jpg",
        "limit":f"{limit}",
        "print_urls":True,
    } 

    print(arguments)
    response = googleimagesdownload()
    response.download(arguments)

parseOptions()
if ('ctg' in opts.keys() and 'kwd' in opts.keys()): main()
