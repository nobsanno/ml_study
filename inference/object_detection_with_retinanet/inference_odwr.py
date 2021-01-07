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
    argparser.add_argument('--img', help=':specify image file path, mov option is precedenced') # use action='store_true' as flag
    argparser.add_argument('--mov', help=':specify movie file path') # use action='store_true' as flag
    argparser.add_argument('--sfn', help=':specify start frame number, default=0') # use action='store_true' as flag
    argparser.add_argument('--ofs', help=':specify output frame size, default=3') # use action='store_true' as flag
    argparser.add_argument('--dbg', help=':debug option', action='store_true') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.dwn: opts.update({'dwn':args.dwn})
    if args.dat: opts.update({'dat':args.dat})
    if args.mdl: opts.update({'mdl':args.mdl})
    if args.img: opts.update({'img':args.img})
    if args.mov: opts.update({'mov':args.mov})
    if args.sfn: opts.update({'sfn':args.sfn})
    if args.ofs: opts.update({'ofs':args.ofs})
    if args.dbg: opts.update({'dbg':args.dbg})

num_classes = 80
image_size = (180, 180)
divopt = False

if __name__ == '__main__':
    parseOptions()
    if ('dwn' in opts.keys()):
        ioc.down_coco2017_ds()

    if ('dat' in opts.keys() and 'mdl' in opts.keys() and ('img' in opts.keys() or 'mov' in opts.keys())):
        datadir = opts['dat']
        mdlfile = opts['mdl']

        if ('img' in opts.keys()): imgfile = opts['img']
        if ('mov' in opts.keys()): movfile = opts['mov']
        else: movfile = False

        if ('sfn' in opts.keys()): start_frame = int(opts['sfn'])
        else: start_frame = 0
        if ('ofs' in opts.keys()): start_frame = int(opts['ofs'])
        else: frame_size = 3
        
        if ('dbg' in opts.keys()): dbgopt = True
        else: dbgopt = False

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

        if (movfile):
            cap = cv2.VideoCapture(movfile)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        count = 0
        while True:
            if (movfile):
                print("frame=" + str(cap.get(cv2.CAP_PROP_POS_FRAMES)) +
                      ", sec=" + str((cap.get(cv2.CAP_PROP_POS_MSEC)/1000)))
                ret, img_array = cap.read()
                if (count > frame_size): ret = False
                simg = img_array
                frame_coun = count + 1
            else:
                if (count == 0): ret = True
                else: ret = False
                fimg = keras.preprocessing.image.load_img(imgfile)
                img_array = keras.preprocessing.image.img_to_array(fimg)
                simg = cv2.imread(imgfile)
                frame_coun = 0
            
            if ret == True:
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
                    simg,
                    image_size,
                    detections.nmsed_boxes[0][:num_detections] / ratio,
                    fc=frame_coun,
                    dbg=dbgopt,
                    div=divopt,
                )
            else:
                break
            
            count = count + 1
