#! /usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["SDL_VIDEO_CENTERED"] = "1"
import matplotlib
matplotlib.use('TkAgg')

import cv2
from core.utils import draw_boxes

import skvideo.io

from skimage import io

io.use_plugin('matplotlib')
from core.xiaofrontend import XiaoYOLO
from core.frontend import YOLO

def main_1():
    LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']
    ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

    # with open(config_path) as config_buffer:
    #    config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(architecture='Full Yolo',
                input_size=416,
                labels=LABELS,
                max_box_per_image=5,
                anchors=ANCHORS)

    ###############################
    #   Load trained weights
    ###############################
    weights_path = './pretrain_models/yolo.weights.h5'
    print(weights_path)
    yolo.load_YOLO_official_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################
    image_path = './images/person.jpg'
    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, LABELS)

    print(len(boxes), 'boxes are found')

    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

    # zebra
    video_inp = '/data/xiao/imagenet/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00005001.mp4'

    # airplane
    video_inp = '/data/xiao/imagenet/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00007011.mp4'

    # # car
    video_inp = '/data/xiao/imagenet/ILSVRC/Data/VID/snippets/val/ILSVRC2015_val_00143003.mp4'

    # cap
    video_inp = '/home/xiao/Downloads/cap3.avi'
    videogen = skvideo.io.vreader(video_inp)
    outputdata = []
    for image in videogen:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes = yolo.predict(image,obj_threshold=0.2, nms_threshold=0.5)

        #filter person out
        boxes = [box for box in boxes if box.get_label()==0]

        image = draw_boxes(image, boxes, labels=LABELS)

        #cv2.imshow('image', image)
        #cv2.waitKey(1)
        image = cv2.resize(image,(0,0),fx=0.5,fy=0.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        outputdata.append(image)

    skvideo.io.vwrite("/home/xiao/Cap/cap3.mp4",outputdata)


def _main_():

    LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']
    ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]



    #with open(config_path) as config_buffer:
    #    config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(architecture='Full Yolo',
                input_size=416,
                labels=LABELS,
                max_box_per_image=5,
                anchors=ANCHORS)

    ###############################
    #   Load trained weights
    ###############################
    weights_path = './pretrain_models/yolo.weights.h5'
    print(weights_path)
    yolo.load_YOLO_official_weights(weights_path)


    # bike
    video_inp = '/home/xiao/Downloads/cap.mp4'
    videogen = skvideo.io.vreader(video_inp,outputdict={'-r': '10'})
    #videogen = skvideo.io.vreader(video_inp)

    i = 0
    for idx,image in enumerate(videogen):

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            boxes = yolo.predict(image, obj_threshold=0.2, nms_threshold=0.5)

            # filter person out
            boxes = [box for box in boxes if box.get_label() == 0]


            for box in boxes:

                xmin = int((box.x - box.w * 1.0 / 2) * image.shape[1])
                xmax = int((box.x + box.w * 1.0 / 2) * image.shape[1])
                ymin = int((box.y - box.h * 1.2 / 2) * image.shape[0])
                ymax = int((box.y + box.h * 0.5 / 2) * image.shape[0])

                #square
                # length = max(box.w/2*image.shape[1], box.h/2*image.shape[0])
                # xmin = int(box.x  * image.shape[1] - length)
                # xmax = int(box.x  * image.shape[1] + length)
                # ymin = int(box.y  * image.shape[0] - length)
                # ymax = int(box.y  * image.shape[0] + length)

                xmin = max(0, xmin)
                xmax = max(0, xmax)
                ymin = max(0, ymin)
                ymax = max(0, ymax)

                xmin = min(image.shape[1] - 1, xmin)
                xmax = min(image.shape[1] - 1, xmax)
                ymin = min(image.shape[0] - 1, ymin)
                ymax = min(image.shape[0] - 1, ymax)

                cv2.imwrite('/home/xiao/Cap/imgs/{:d}.jpg'.format(i),image[ymin:ymax, xmin:xmax, :])
                i += 1


if __name__ == '__main__':
    main_1()