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
from preprocessing import *
from sklearn.externals import joblib
from metric import evaluator
def _main_():


    #training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_voc2007_3'
    training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_caltech_3'
    gen_dataset = parse_annotation_voc
    best_only = False

    ###############################
    #   Load config
    ###############################
    config_path = os.path.join(training_result_folder, 'config.json')
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)


    ###############################
    #   Construct the model
    ###############################
    yolo = YOLO.init_from_config(config)



    ###############################
    #   Load the pretrained weights (if any)
    ###############################
    model_path = 'best_' + config['train']['saved_weights_name'] if best_only else config['train']['saved_weights_name']
    #validation_model_path = os.path.join(training_result_folder,  'best_' + config['train']['saved_weights_name'] )
    validation_model_path = os.path.join(training_result_folder, model_path)
    if os.path.exists(validation_model_path):
        print("Loading pre-trained weights in", validation_model_path)
        yolo.load_weights(validation_model_path)
    else:
        raise FileNotFoundError('cannot find model: %s' % validation_model_path)




    ###############################
    #  Load validation set
    ###############################
        # parse annotations of the validation set, if any, otherwise split the training set
    if ('valid_annot_file' in config['valid'].keys() and os.path.exists(config['valid']['valid_annot_file'])):
        print("Reading val annotations...")
        valid_imgs, valid_labels = joblib.load(config['valid']['valid_annot_file'])
    elif os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = gen_dataset(config['valid']['valid_annot_folder'],
                                               config['valid']['valid_image_folder'],
                                               config['model']['labels'])
    else:
        raise FileNotFoundError('cannot load validation set')


    ###############################
    #   perform evaluation
    ###############################
    eval_folder = os.path.join(training_result_folder, 'evaluation')
    os.makedirs(eval_folder, exist_ok=True)
    print('Evaluating...', config['model']['labels'])

    result_file = 'best_result_dict.pkl' if best_only else 'result_dict.pkl'
    if os.path.exists(os.path.join(eval_folder, result_file)) == False:
        result = evaluator.evaluate(valid_imgs,yolo,config,0.5)
        joblib.dump(result,os.path.join(eval_folder, result_file))
    else:
        result = joblib.load(os.path.join(eval_folder, result_file))

    evaluator.sumnmarize_result(result,config['model']['labels'],eval_folder)

if __name__ == '__main__':

    _main_()