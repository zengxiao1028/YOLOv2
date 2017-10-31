import os
import numpy as np
from preprocessing import parse_annotation_voc
from frontend import YOLO
import json
import skvideo.io
import tqdm
import cv2
from utils import draw_boxes
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from moviepy.editor import *

def _main_():


    config_path = './exp_configs/config.json'

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)


    ###############################
    #   Construct the model
    ###############################

    yolo = YOLO.init_from_config(config)

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])


    video_inp = '/data/xiao/imagenet/ILSVRC/Data/VID/snippets/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00010001.mp4'
    video_out = './result.mp4'

    metadata = skvideo.io.ffprobe(video_inp)
    video_height = metadata["video"]["@height"]
    video_width = metadata["video"]["@width"]
    num_frames = metadata["video"]["@nb_frames"]

    videogen = skvideo.io.vreader(video_inp)
    outputdata = []

    for image in videogen:

        boxes = yolo.predict(image,0.1,0.1)

        image = draw_boxes(image, boxes, labels=config['model']['labels'])

        outputdata.append(image)


    skvideo.io.vwrite(video_out, np.array(outputdata).astype(np.uint8))
    clip = VideoFileClip(video_out)
    clip.preview()


if __name__ == '__main__':

    _main_()