from argparse import ArgumentParser
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import cv2
import inference_odwr_com as ioc

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--dwn', help=':preparing data set', action='store_true') # use action='store_true' as flag
    argparser.add_argument('--dat', help=':specify data dir path') # use action='store_true' as flag
    argparser.add_argument('--mdl', help=':specify model file path') # use action='store_true' as flag
    argparser.add_argument('--mov', help=':specify movie file path') # use action='store_true' as flag
    argparser.add_argument('--sfn', help=':specify start frame number, default=0') # use action='store_true' as flag
    argparser.add_argument('--ofs', help=':specify output frame size, default=3') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.dwn: opts.update({'dwn':args.dwn})
    if args.dat: opts.update({'dat':args.dat})
    if args.mdl: opts.update({'mdl':args.mdl})
    if args.mov: opts.update({'mov':args.mov})
    if args.sfn: opts.update({'sfn':args.sfn})
    if args.ofs: opts.update({'ofs':args.ofs})

num_classes = 80
image_size = (150, 150)
divopt = True
classes = ["dog", "cat", "giraffe", "elephant", "lion"]
frame_skip = 14

if __name__ == '__main__':
    parseOptions()
    if ('dwn' in opts.keys()):
        ioc.down_coco2017_ds()

    if ('dat' in opts.keys() and 'mdl' in opts.keys() and 'mov' in opts.keys()):
        datadir = opts['dat']
        mdlfile = opts['mdl']
        movfile = opts['mov']

        if ('sfn' in opts.keys()): start_frame = int(opts['sfn'])
        else: start_frame = 0
        if ('ofs' in opts.keys()): frame_size = int(opts['ofs'])
        else: frame_size = 3
        
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

        cap = cv2.VideoCapture(movfile)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        count = 0
        while True:
            ret, img_array = cap.read()
            if (count > frame_size): ret = False
            
            if ret == True:
                print("frame=" + str(cap.get(cv2.CAP_PROP_POS_FRAMES)-1) +
                      ", sec=" + str((cap.get(cv2.CAP_PROP_POS_MSEC)/1000)))
                image = tf.cast(img_array, dtype=tf.float32)
                input_image, ratio = ioc.prepare_image(image)

                detections = inference_model.predict(input_image)
                num_detections = detections.valid_detections[0]
                class_names = [ int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections] ]

                ioc.second_classification_to_vs(
                    mdlfile,
                    img_array,
                    image_size,
                    detections.nmsed_boxes[0][:num_detections] / ratio,
                    classes,
                    fp=(cap.get(cv2.CAP_PROP_POS_FRAMES)-1),
                    div=divopt,
                )
            else:
                break
            
            count = count + 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, (cap.get(cv2.CAP_PROP_POS_FRAMES)+frame_skip))
