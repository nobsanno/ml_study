from argparse import ArgumentParser
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--wmn', help=':specify write model name') # use action='store_true' as flag
    argparser.add_argument('--ncl', help=':specify number of class') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.wmn: opts.update({'wmn':args.wmn})
    if args.ncl: opts.update({'ncl':args.ncl})

IMG_SIZE = 224

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    return keras.Model(inputs, outputs, name="EfficientNet")

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

if __name__ == '__main__':
    parseOptions()
    if ('wmn' in opts.keys() and 'ncl' in opts.keys()):
        num_classes = int(opts['ncl'])
        model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes)
        unfreeze_model(model)
        model.summary()
        model.save(opts['wmn'])
