import numpy as np
import os
import xml.etree.ElementTree as ET
import random
import cv2
import warnings
import skvideo.io
from sklearn.externals import joblib
import skvideo.utils

import json
import ffmpeg

def read_gt(gt_path):
    gt_dict = {}
    with open(gt_path) as f:
        lines = f.read().splitlines()
        for line in lines:
            items = line.split(',')
            frame = int(items[0])
            nb_objects = int(items[1])
            objects = []
            for i in range(nb_objects):
                c = items[6 + 5 * i]

                if c == 'null':
                    continue
                object = {}
                x = int(items[2 + 5 * i])
                y = int(items[3 + 5 * i])
                w = int(items[4 + 5 * i])
                h = int(items[5 + 5 * i])

                object['xmin'] = x
                object['ymin'] = y
                object['xmax'] = x + w
                object['ymax'] = y + h
                object['name'] = c

                objects.append(object)
            if len(objects) > 0:
                gt_dict[frame] = objects
    return gt_dict

def main():
    video_folder = '/home/xiao/Cap/videos'
    label_folder = '/home/xiao/Cap/self_label/video_label/video/results'
    save_folder = '/home/xiao/Cap/data/imgs'
    video_files = os.listdir(video_folder)
    imgs = []
    for file in video_files:
        if os.path.isdir(os.path.join(video_folder,file)):
            continue
        gt_file = file[:-4] + '_gt.txt'
        gt_path = os.path.join(label_folder,gt_file)
        if not os.path.exists(gt_path):
            continue
        video_path = os.path.join(video_folder,file)


        videogen = skvideo.io.vreader(video_path)

        videodata = skvideo.io.ffprobe(video_path)
        W , H = videodata['video']['@width'], videodata['video']['@height']

        gt_dict = read_gt(gt_path)

        img_save_folder = os.path.join(save_folder,file[17:-4])
        os.makedirs(img_save_folder,exist_ok=True)
        for idx,image in enumerate(videogen):
            img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            if idx % 6 ==0 :
                print(idx)
                if idx in gt_dict:
                    sample_dict = {}
                    filename = os.path.join(img_save_folder, str(idx) + '.jpg')
                    sample_dict['filename'] = filename
                    sample_dict['height'] = H
                    sample_dict['width'] = W
                    sample_dict['object'] = gt_dict[idx]
                    cv2.imwrite(filename, img)
                    imgs.append(sample_dict)

    random.shuffle(imgs)
    labels = {'pass','fail'}
    joblib.dump((imgs,labels), './train.pkl')
    print(len(imgs))

if __name__ == '__main__':
    #main()
    imgs,labels = joblib.load('./train.pkl')
    print(len(imgs))