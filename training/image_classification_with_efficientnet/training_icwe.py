from argparse import ArgumentParser
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--rmn', help=':specify read model name') # use action='store_true' as flag
    argparser.add_argument('--wmn', help=':specify write model name') # use action='store_true' as flag
    argparser.add_argument('--img', help=':specify image dir path') # use action='store_true' as flag
    argparser.add_argument('--epc', help=':specify number of epoch') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.rmn: opts.update({'rmn':args.rmn})
    if args.wmn: opts.update({'wmn':args.wmn})
    if args.img: opts.update({'img':args.img})
    if args.epc: opts.update({'epc':args.epc})

IMG_SIZE = 224
# batch_size = 64
batch_size = 48

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

if __name__ == '__main__':
    parseOptions()
    if (not ('rmn' in opts.keys() and 'wmn' in opts.keys() and 'img' in opts.keys() and 'epc' in opts.keys())):
        sys.exit()

    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        opts['img'],
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
    )

    ds_test = tf.keras.preprocessing.image_dataset_from_directory(
        opts['img'],
        labels='inferred',
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
    )

    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    epochs = int(opts['epc'])

    model = load_model(opts['rmn'])
    model.summary()
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
    )
    hist = model.fit(ds_train, validation_data=ds_test, epochs=epochs, verbose=2)
    model.save(opts['wmn'])
    plot_hist(hist)
