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
    argparser.add_argument('--prp', help=':preparing data set', action='store_true') # use action='store_true' as flag
    argparser.add_argument('--fil', help=':filtering data set', action='store_true') # use action='store_true' as flag
    argparser.add_argument('--rmn', help=':specify read model name') # use action='store_true' as flag
    argparser.add_argument('--wmn', help=':specify write model name') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.prp: opts.update({'prp':args.prp})
    if args.fil: opts.update({'fil':args.fil})
    if args.rmn: opts.update({'rmn':args.rmn})
    if args.wmn: opts.update({'wmn':args.wmn})

image_size = (180, 180)
batch_size = 32
epochs = 50

parseOptions()

"""
## Load the data: the Cats vs Dogs dataset
### Raw data download
First, let's download the 786M ZIP archive of the raw data:
"""

"""shell
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
"""

"""shell
unzip -q kagglecatsanddogs_3367a.zip
"""

"""
Now we have a `PetImages` folder which contain two subfolders, `Cat` and `Dog`. Each
 subfolder contains image files for each category.
"""

if ('prp' in opts.keys()):
    os.system(r'curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip')
    os.system(r'unzip -q kagglecatsanddogs_3367a.zip')

"""
### Filter out corrupted images
When working with lots of real-world image data, corrupted images are a common
occurence. Let's filter out badly-encoded images that do not feature the string "JFIF"
 in their header.
"""

if ('fil' in opts.keys()):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

if (not ('rmn' in opts.keys() and 'wmn' in opts.keys())):
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
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
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

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

model = load_model(opts['rmn'])
model.summary()
model.compile(
    loss="binary_crossentropy",
    metrics=["accuracy"],
    optimizer=keras.optimizers.Adam(1e-3),
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

"""
We get to ~96% validation accuracy after training for 50 epochs on the full dataset.
"""

model.save(opts['wmn'])
