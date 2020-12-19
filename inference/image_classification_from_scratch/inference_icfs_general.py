from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import cv2

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

def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[1, 0, 0]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)

    f.add_subplot(1, n, i + 1)
    i = i + 1
    plt.title('original')
    plt.axis("off")
    plt.imshow(image)

def classification(
     mdlfile, imgdir, figsize=(10, 7)
):
    model = load_model(mdlfile)
    model.summary()

    image_files = glob.glob(f"{imgdir}/**", recursive=True)

    size = int(len(image_files))
    width = 5
    if (size < width): width = size
    height = (size / width) + (size % width)
    f = plt.figure(figsize=figsize)
    i = 0

    for image_file in image_files:
        m = re.search(r'\.jpg$', image_file)
        if (m):
            print(image_file)

            img = keras.preprocessing.image.load_img(image_file, target_size=image_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array)

            f.add_subplot(width, height, i + 1)
            i = i + 1
            plt.title(float("{:.5f}".format(predictions[0][0])))
            plt.axis("off")
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
    
    plt.show()

parseOptions()
if ('mdl' in opts.keys() and 'img' in opts.keys()):
    mdlfile = opts['mdl']
    imgdir = opts['img']

    classification(
        mdlfile,
        imgdir,
    )
