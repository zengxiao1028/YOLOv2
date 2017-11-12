import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from frontend import YOLO
import json
from xiaofrontend import XiaoYOLO

from preprocessing import *
from sklearn.externals import joblib
from metric import evaluator
def _main_():

    training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_voc2007_7'
    #training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_imagenetvid_4'
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
    eval_folder = os.path.join(training_result_folder, 'best_only_evaluation' if best_only else 'evaluation')
    os.makedirs(eval_folder, exist_ok=True)
    print('Evaluating...', config['model']['labels'])

    result_file = 'result_dict.pkl' # stores result and map
    prediction_file = 'prediction_dict.pkl' # stores prediction boxes
    if os.path.exists(os.path.join(eval_folder, result_file)) == False:
        result = evaluator.evaluate(valid_imgs,yolo,config,iou_threshold=0.5, obj_threshold=config['valid']['obj_threshold'], nms_threshold=config['valid']['nms_threshold'])
        result_dict = (result[0],result[1])
        prediction_dict = result[2]
        joblib.dump( result_dict, os.path.join(eval_folder, result_file))
        joblib.dump(  prediction_dict, os.path.join(eval_folder, prediction_file))
    else:
        result_dict = joblib.load(os.path.join(eval_folder, result_file))

    evaluator.sumnmarize_result(result_dict,config['model']['labels'],eval_folder)

if __name__ == '__main__':

    _main_()