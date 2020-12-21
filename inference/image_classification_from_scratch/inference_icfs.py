from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import re
import os

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--mdl', help=':specify model file name') # use action='store_true' as flag
    argparser.add_argument('--img', help=':specify image dir path') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.mdl: opts.update({'mdl':args.mdl})
    if args.img: opts.update({'img':args.img})

image_size = (180, 180)
ext = r'\.jpg$'

def filter(
     imgdir
):
    image_files = glob.glob(f"{imgdir}/**", recursive=True)

    for image_file in image_files:
        m = re.search(ext, image_file)
        if (m):
            try:
                fobj = open(image_file, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                print(f"Error: {image_file} can not opend!")
                # Delete corrupted image
                os.remove(image_file)

def classification(
     mdlfile, imgdir, figsize=(15, 9)
):
    model = load_model(mdlfile)
    model.summary()

    image_files = glob.glob(f"{imgdir}/**", recursive=True)

    size = int(len(image_files))
    width = 5
    if (size < width): width = size
    height = (size / width)
    if (size % width): height = height + 1
    f = plt.figure(figsize=figsize)
    i = 0

    for image_file in image_files:
        m = re.search(ext, image_file)
        if (m):
            img = keras.preprocessing.image.load_img(image_file, target_size=image_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array)
            print(f"{image_file}:{predictions}")

            f.add_subplot(width, height, i + 1)
            i = i + 1
            plt.title(float("{:.2f}".format(predictions[0][0])))
            plt.axis("off")
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
    
    plt.show()

parseOptions()
if ('mdl' in opts.keys() and 'img' in opts.keys()):
    mdlfile = opts['mdl']
    imgdir = opts['img']

    filter(imgdir)
    classification(mdlfile, imgdir)
