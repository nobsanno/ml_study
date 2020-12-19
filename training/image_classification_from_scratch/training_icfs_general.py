from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
import sys

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--rmn', help=':specify read model name') # use action='store_true' as flag
    argparser.add_argument('--wmn', help=':specify write model name') # use action='store_true' as flag
    argparser.add_argument('--img', help=':specify image dir path') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.rmn: opts.update({'rmn':args.rmn})
    if args.wmn: opts.update({'wmn':args.wmn})
    if args.img: opts.update({'img':args.img})

image_size = (180, 180)
batch_size = 32
epochs = 50

parseOptions()

if (not ('rmn' in opts.keys() and 'wmn' in opts.keys() and 'img' in opts.keys())):
    sys.exit()

"""
## Standardizing the data
Our image are already in a standard size (180x180), as they are being yielded as
contiguous `float32` batches by our dataset. However, their RGB channel values are in
 the `[0, 255]` range. This is not ideal for a neural network;
in general you should seek to make your input values small. Here, we will
standardize values to be in the `[0, 1]` by using a `Rescaling` layer at the start of
 our model.
"""

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    opts['img'],
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    opts['img'],
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

"""
## Configure the dataset for performance
Let's make sure to use buffered prefetching so we can yield data from disk without
 having I/O becoming blocking:
"""

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

"""
## Train the model
"""

model = load_model(opts['rmn'])
model.summary()
model.compile(
    loss="binary_crossentropy",
    metrics=["accuracy"],
    optimizer=keras.optimizers.Adam(1e-3),
)
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

"""
We get to ~96% validation accuracy after training for 50 epochs on the full dataset.
"""

model.save(opts['wmn'])
