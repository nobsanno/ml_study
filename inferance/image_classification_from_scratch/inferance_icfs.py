from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--mdl', help=':specify model file path') # use action='store_true' as flag
    argparser.add_argument('--img', help=':specify image file path') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.mdl: opts.update({'mdl':args.mdl})
    if args.img: opts.update({'img':args.img})

# image_size = (180, 180)
image_size = (150, 150)

def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[1, 0, 0]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        print(f"predictions={_cls}, {score}")
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

parseOptions()
if ('mdl' in opts.keys() and 'img' in opts.keys()):
    modelfile = opts['mdl']
    imgfile = opts['img']

    model = load_model(modelfile)
    model.summary()

    img = keras.preprocessing.image.load_img(
        imgfile, target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]

    classes = list(range(0))
    if ((100 * score) >= 50):
        classes.append('dog')
    else:
        classes.append('cat')
        score = (1 - score)

    image = plt.imread(imgfile)
    visualize_detections(
        image,
        [[10, 10, 110, 110]],
        classes,
        score,
    )
