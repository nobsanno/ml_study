from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import inference_odwr_com as ioc

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--dwn', help=':downloading data set', action='store_true') # use action='store_true' as flag
    argparser.add_argument('--dat', help=':specify data dir path') # use action='store_true' as flag
    argparser.add_argument('--mdl', help=':specify model file path') # use action='store_true' as flag
    argparser.add_argument('--img', help=':specify image file path') # use action='store_true' as flag
    argparser.add_argument('--dbg', help=':debug option', action='store_true') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.dwn: opts.update({'dwn':args.dwn})
    if args.dat: opts.update({'dat':args.dat})
    if args.mdl: opts.update({'mdl':args.mdl})
    if args.img: opts.update({'img':args.img})
    if args.dbg: opts.update({'dbg':args.dbg})

num_classes = 80
image_size = (180, 180)

if __name__ == '__main__':
    parseOptions()
    if ('dwn' in opts.keys()):
        ioc.down_coco2017_ds()

    if ('dat' in opts.keys() and 'mdl' in opts.keys() and 'img' in opts.keys()):
        datadir = opts['dat']
        mdlfile = opts['mdl']
        imgfile = opts['img']
        dbgopt = False
        if ('dbg' in opts.keys()): dbgopt = True

        resnet50_backbone = ioc.get_backbone()
        model = ioc.RetinaNet(num_classes, resnet50_backbone)

        latest_checkpoint = tf.train.latest_checkpoint(datadir)
        model.load_weights(latest_checkpoint)

        val_dataset, dataset_info = tfds.load("coco/2017", split="validation", with_info=True, data_dir=datadir)
        int2str = dataset_info.features["objects"]["label"].int2str

        image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = model(image, training=False)
        detections = ioc.DecodePredictions(confidence_threshold=0.5)(image, predictions)
        inference_model = tf.keras.Model(inputs=image, outputs=detections)

        img = keras.preprocessing.image.load_img(imgfile)
        img_array = keras.preprocessing.image.img_to_array(img)
        image = tf.cast(img_array, dtype=tf.float32)
        input_image, ratio = ioc.prepare_image(image)

        detections = inference_model.predict(input_image)
        num_detections = detections.valid_detections[0]
        class_names = [ int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections] ]

        if (dbgopt):
            ioc.visualize_detections(
                image,
                detections.nmsed_boxes[0][:num_detections] / ratio,
                class_names,
                detections.nmsed_scores[0][:num_detections],
            )

        ioc.second_classification(
            mdlfile,
            imgfile,
            image_size,
            detections.nmsed_boxes[0][:num_detections] / ratio,
            dbg=dbgopt,
        )
