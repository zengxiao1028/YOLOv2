import numpy as np
import os
import xml.etree.ElementTree as ET
import random
import cv2
import warnings

from sklearn.externals import joblib



def main():
    video_folder = '/home/xiao/Cap/self_label/video_label/video'
    label_folder = '/home/xiao/Cap/self_label/video_label/video/results'
    video_files = os.listdir(video_folder)
    for file in video_files:
        gt_file = file[:-4] + '.txt'
        video_path = os.path
        videogen = skvideo.io.vreader(video_inp)
        #get gt

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
                    elif c == 'pass':
                        c = 1
                    elif c == 'fail':
                        c = 2

                    x = int(items[2 + 5 * i]) / W
                    y = int(items[3 + 5 * i]) / H
                    w = int(items[4 + 5 * i]) / W
                    h = int(items[5 + 5 * i]) / H

                    objects.append((x, y, w, h, c))
                if len(objects) > 0:
                    gt_dict[frame] = objects



if __name__ == '__main__':