import json
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from core.frontend import YOLO
import skvideo.io
import cv2
import numpy as np
from core.preprocessing import parse_annotation_voc
from core.utils import draw_boxes


from core.xiaofrontend import XiaoYOLO
from sklearn.externals import joblib
def _main_():

    training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_cap_2'
    #training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_imagenetvid_7'
    #training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_caltech_2'

    gen_dataset_fn = parse_annotation_voc


    config_path = os.path.join(training_result_folder, 'config.json')
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)



    #validation_model_path = os.path.join(training_result_folder,  'best_' + config['train']['saved_weights_name'] )
    validation_model_path = os.path.join(training_result_folder, config['train']['saved_weights_name'])
    #validation_model_path = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_cap_2/full_yolo_004.h5'



    ###############################
    #   Construct the model
    ###############################

    yolo = YOLO.init_from_config(config)

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    if os.path.exists(validation_model_path):
        print("Loading pre-trained weights in", validation_model_path)
        yolo.load_weights(validation_model_path)
        #yolo.load_model(validation_model_path)


    ###############################
    #   Predict validation set
    ###############################
    # if ('valid_annot_file' in config['valid'].keys() and os.path.exists(config['valid']['valid_annot_file'])):
    #     print("Reading val annotations...")
    #     valid_imgs, valid_labels = joblib.load(config['valid']['valid_annot_file'])
    # elif os.path.exists(config['valid']['valid_annot_folder']):
    #     valid_imgs, valid_labels = gen_dataset_fn(config['valid']['valid_annot_folder'],
    #                                                     config['valid']['valid_image_folder'],
    #                                                     config['model']['labels'])
    #
    # for sample in valid_imgs:
    #     image = cv2.imread(sample['filename'])
    #     boxes = yolo.predict(image, obj_threshold=0.2, nms_threshold=0.3)
    #
    #     image = draw_boxes(image, boxes, labels=config['model']['labels'])
    #
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)



    ###############################
    #  Predict image
    ###############################
    # image_path = '/home/xiao/video_project/YOLOv2/dataset/bloodcell/JPEGImages/BloodImage_00351.jpg'
    # image = cv2.imread(image_path)
    # boxes = yolo.predict(image,0.5,0.3)
    # image = draw_boxes(image, boxes, config['model']['labels'])
    #
    # print(len(boxes), 'boxes are found')
    #
    # cv2.imwrite('./tmp/result.jpg', image)



    ###############################
    #   Predict video
    ###############################

    #video_inp = '/home/xiao/video_project/YOLOv2/dataset/caltech_pedestrian/data/plots/set07_V002.avi'
    video_inp = '/home/xiao/Cap/videos/2.avi'
    video_out = './tmp/result.mp4'

    metadata = skvideo.io.ffprobe(video_inp)


    videogen = skvideo.io.vreader(video_inp)
    outputdata = []

    for idx,image in enumerate(videogen):

        if idx%3==0:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            boxes = yolo.predict(image, 0.3, 0.5,['pass'])

            image = draw_boxes(image, boxes, labels=sorted(list(config['model']['labels'])) )

            cv2.imshow('image', image)
            cv2.waitKey(1)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            outputdata.append(image)


    skvideo.io.vwrite(video_out, np.array(outputdata).astype(np.uint8))
    # clip = VideoFileClip(video_out)
    # clip.preview()






if __name__ == '__main__':

    _main_()