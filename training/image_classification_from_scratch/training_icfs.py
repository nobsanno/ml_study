from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras.models import load_model
import training_icfs_com as tic

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

if __name__ == '__main__':
    parseOptions()
    if ('rmn' in opts.keys() and 'wmn' in opts.keys() and 'img' in opts.keys()):
        (train_ds, val_ds) = tic.prepare_train_data(opts['img'], image_size, batch_size, sigen=False)

        model = load_model(opts['rmn'])
        model.summary()
        model.compile(
            loss="binary_crossentropy",
            metrics=["accuracy"],
            optimizer=keras.optimizers.Adam(1e-3),
        )
        model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        model.save(opts['wmn'])
