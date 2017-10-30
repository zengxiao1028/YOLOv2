#! /usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
import os
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



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

    weights_path = './pretrain_models/yolo.weights.h5'
    image_path = './images/person.jpg'

    #with open(config_path) as config_buffer:
    #    config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(architecture='Full Yolo',
                input_size=416,
                labels=LABELS,
                max_box_per_image=10,
                anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])

    ###############################
    #   Load trained weights
    ###############################
    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes
    ###############################

    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, LABELS)

    print(len(boxes), 'boxes are found')

    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


if __name__ == '__main__':
    _main_()