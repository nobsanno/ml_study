from argparse import ArgumentParser
import tensorflow as tf
import cv2
import glob
import os
import sys

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--img', help=':specify image dir') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.img: opts.update({'img':args.img})

def get_resolution(filepath):
    img = cv2.imread(filepath)
 
    if img is None:
        print("Failed to load image file.")
        sys.exit(1)
 
    if len(img.shape) == 3:
        height, width, channels = img.shape[:3]
    else:
        height, width = img.shape[:2]
        channels = 1
    
    return width,height,channels

def is_empty(directory):
    files = os.listdir(directory)
    files = [f for f in files if not f.startswith(".")]
    if not files:
        return True
    else:
        return False

def rm_min_img(imgdir, min_width=500, min_height=500):
    targets = glob.glob(f"{imgdir}/**", recursive=True)

    for tgt in targets:
        if (os.path.isfile(tgt)):
            try:
                fobj = open(tgt, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                print(f"Error: {tgt} can not opend!")
                # Delete corrupted image
                os.remove(tgt)
            else:
                width,height,channels = get_resolution(tgt)
                if (width < min_width or height < min_height):
                    print(f"{tgt} = {width} x {height}, so removed.")
                    os.remove(tgt)

        elif (os.path.isdir(tgt)):
            if (is_empty(tgt)):
                print(f"{tgt} is empty, so removed.")
                os.rmdir(tgt)

parseOptions()
if ('img' in opts.keys()):
    rm_min_img(opts['img'])
