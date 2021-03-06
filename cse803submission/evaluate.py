import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2
import numpy as np
from core.utils import draw_boxes
from core.frontend import YOLO
import json
import time
from collections import defaultdict
from keras.preprocessing.image import load_img
def main(eval_folder='/home/xiao/video_project/YOLOv2/cse803submission/image_comdata', online_prediction=False):


    labels = ["apple", "banana", "broccoli", "burger", "cookie", "egg", "frenchfry", "hotdog", "pasta", "pizza", "rice", "salad", "strawberry", "tomato"]


    label_file = os.path.join(eval_folder,'label.txt')
    prediction_file = os.path.join(eval_folder, 'prediction.txt')

    if os.path.exists(label_file) is False:
        raise FileNotFoundError()

    with open(label_file) as f:
        lines = f.read().splitlines()
        ground_truth_dict = { line.split(' ')[0]:line.split(' ')[1:] for line in lines }

    if online_prediction:
        training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_meal_8'
        config_path = os.path.join(training_result_folder, 'config.json')
        with open(config_path) as config_buffer:
            config = json.load(config_buffer)
        yolo = YOLO.init_from_config(config)
        #validation_model_path = os.path.join(training_result_folder, config['train']['saved_weights_name'])
        validation_model_path = os.path.join(training_result_folder, 'best_full_yolo.h5')
        #validation_model_path = os.path.join(training_result_folder, 'full_yolo_014.h5')
        if os.path.exists(validation_model_path):
            print("Loading pre-trained weights in", validation_model_path)
            yolo.load_weights(validation_model_path)

        results_dict = dict()
        time_used = []
        for k, v in ground_truth_dict.items():
            if os.path.exists(os.path.join(eval_folder,k)) is False:
                k1 = k[:-3] + 'JPG'
            # img = load_img(os.path.join(eval_folder,k1))
            # img = np.array(img)
            # img = img[...,::-1]
            else:
                k1 = k
            img = cv2.imread(os.path.join(eval_folder,k1))
            if img is None:
                print('Evaluating %s' % os.path.join(eval_folder, k))
                print(img.shape)

            # get predictions
            time1 = time.time()
            boxes = yolo.predict(img, obj_threshold=0.01, nms_threshold=0.3)
            time2 = time.time()
            time_used.append(time2-time1)
            # filter
            predictions = filter_prediction(boxes, 0.35)

            #visualize it
            image = draw_boxes(img, predictions, labels=config['model']['labels'])
            cv2.imwrite(os.path.join(os.path.join(eval_folder,'prediction'),k[:-4]+'_out.jpg'),image)

            # transform output to labels
            predictions = [labels[prediction.get_label()] for prediction in predictions]
            results_dict[k] = predictions

    else:
        with open(prediction_file) as f:
            lines = f.read().splitlines()
            results_dict = { line.split(' ')[0]:line.split(' ')[1:] for line in lines }

    print('average_time:', np.mean(time_used))
    #compute recogntion rate = 0.5 * (detection rate + rejection rate)
    tp_dict = defaultdict(int)

    fp_dict = defaultdict(int)

    for k, v in ground_truth_dict.items():
        # compute detection rate
        #for every l1 in ground_truth labels
        for l1 in v:
            #l successfully detected
            if l1 in results_dict[k]:
                tp_dict[l1] = tp_dict[l1] + 1

            # # l not detected
            # else:
            #     fn_dict[l1] = fn_dict[l1] + 1

        # compute rejection rate
        # for every l2 in predictions
        for l2 in results_dict[k]:
            if l2 in v:
                pass

            # l2 miss fire
            else:
                fp_dict[l2] = fp_dict[l2] + 1


    items = ['salad','pasta','hotdog','frenchfry','burger','apple','banana','broccoli','pizza','egg','tomato','rice','strawberry','cookie']
    recognition_rates = []
    for cls in items:
        # compute detection rate
        cls_images= [ k for k, v in ground_truth_dict.items() if cls in v ]
        p_images = len(cls_images)
        detection_rate = tp_dict[cls] * 1.0 / p_images

        n_images = len(ground_truth_dict.items()) - p_images
        rejection_rate = 1 - fp_dict[cls] * 1.0 / n_images
        print('%s : detection rate: %.2f, reject rate: % 0.2f, recogntion rate %.2f '
              % (cls,detection_rate,rejection_rate,(detection_rate+rejection_rate)/2.))
        recognition_rates.append((detection_rate+rejection_rate)/2.)

    print(np.mean(recognition_rates),np.sum(recognition_rates))


def filter_prediction(boxes, sub_threshold = 0.5):

    boxes = sorted(boxes, key=lambda box: box.get_score(),reverse=True)

    predictions = [ boxes[0] ]

    for i in range(len(boxes)):
        box = boxes[i]
        if box.get_score() > sub_threshold:
            predictions.append(box)

    return predictions

if __name__ == '__main__':
    main(online_prediction=True)