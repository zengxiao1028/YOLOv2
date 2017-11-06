import os
import re
import json
import glob
import cv2 as cv
from collections import defaultdict
from sklearn.externals import joblib

def main(caltech_pedestrian_data_folder='./'):

    #annotation_file = os.path.join(caltech_pedestrian_data_folder,'annotations.json')
    annotation_file = '/Users/xiaozeng/Downloads/annotations.json'
    annotations = json.load(open(annotation_file))

    out_dir = 'data/plots'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_fns = defaultdict(dict)

    for fn in sorted(glob.glob(os.path.join(caltech_pedestrian_data_folder,'images/*.png'))):
        set_name = re.search('(set[0-9]+)', fn).groups()[0]
        img_fns[set_name] = defaultdict(dict)

    for fn in sorted(glob.glob(os.path.join(caltech_pedestrian_data_folder,'images/*.png'))):
        set_name = re.search('(set[0-9]+)', fn).groups()[0]
        video_name = re.search('(V[0-9]+)', fn).groups()[0]
        img_fns[set_name][video_name] = []

    for fn in sorted(glob.glob(os.path.join(caltech_pedestrian_data_folder,'images/*.png'))):
        set_name = re.search('(set[0-9]+)', fn).groups()[0]
        video_name = re.search('(V[0-9]+)', fn).groups()[0]
        n_frame = re.search('_([0-9]+)\.png', fn).groups()[0]
        img_fns[set_name][video_name].append((int(n_frame), fn))

    train = []
    val = []
    for set_name in sorted(img_fns.keys()):
        for video_name in sorted(img_fns[set_name].keys()):
            for frame_i, fn in sorted(img_fns[set_name][video_name]):

                objects = []

                if str(frame_i) in annotations[set_name][video_name]['frames']:
                    data = annotations[set_name][
                        video_name]['frames'][str(frame_i)]
                    for datum in data:
                        x, y, w, h = [int(v) for v in datum['pos']]
                        objects.append({'xmin': x, 'xmax': x + w, 'ymin': y, 'ymax': y + h, 'name': 'pedestrian'})

                if len(objects) > 0 :
                    sample = {'object':objects,'filename':fn,'height':480,'width':640,'set_name':set_name,'video_name':video_name}
                    #training set
                    if 0 <= int(set_name[-2:]) <= 5:
                        train.append(sample)
                    #validation set
                    elif 6 <= int(set_name[-2:]) <= 10:
                        val.append(sample)
                    else:
                        raise ValueError('Wrong set name: %s' % set_name)
            print(set_name, video_name)

    label = {'pedestrian'}
    joblib.dump((train,label), os.path.join(caltech_pedestrian_data_folder,'train.pkl'))
    joblib.dump((val, label), os.path.join(caltech_pedestrian_data_folder, 'val.pkl'))

if __name__ == '__main__':
    main()