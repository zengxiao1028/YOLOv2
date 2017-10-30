import os
import numpy as np
from preprocessing import parse_annotation_voc
from frontend import YOLO
import json
from sklearn.externals import joblib
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def _main_():

    config_path = './exp_configs/vid_config.json'

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Parse the annotations 
    ###############################

    # # parse annotations of the training set
    # train_imgs, train_labels = parse_annotation_voc(config['train']['train_annot_folder'],
    #                                                 config['train']['train_image_folder'],
    #                                                 config['model']['labels'])
    #
    # # parse annotations of the validation set, if any, otherwise split the training set
    # if os.path.exists(config['valid']['valid_annot_folder']):
    #     valid_imgs, valid_labels = parse_annotation_voc(config['valid']['valid_annot_folder'],
    #                                                     config['valid']['valid_image_folder'],
    #                                                     config['model']['labels'])
    # else:
    #     train_valid_split = int(0.8*len(train_imgs))
    #     np.random.shuffle(train_imgs)
    #
    #     valid_imgs = train_imgs[train_valid_split:]
    #     train_imgs = train_imgs[:train_valid_split]

    print("Reading train annotations...")
    train_imgs, train_labels = joblib.load(config['train']['train_annot_file'])
    print("Reading val annotations...")
    valid_imgs, valid_labels = joblib.load(config['valid']['valid_annot_file'])


    if len(set(config['model']['labels']).intersection(train_labels)) == 0:
        print("Labels to be detected are not present in the dataset! Please revise the list of labels in the config.json file!")
        
        return

    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process 
    ###############################

    yolo.train(config_path        = config_path,
               train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epoch           = config['train']['nb_epoch'], 
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               warmup_bs          = config['train']['warmup_batches'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               name               = config['train']['name'],
               debug              = config['train']['debug'])

if __name__ == '__main__':

    _main_()