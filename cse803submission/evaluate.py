import numpy as np
import os
import cv2
from core.frontend import YOLO
import json
def main(eval_folder='./imaged_tryout'):

    training_result_folder = '/home/xiao/video_project/YOLOv2/traning_results/YOLOv2_meal_6'
    config_path = os.path.join(training_result_folder, 'config.json')
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    yolo = YOLO.init_from_config(config)


    label_file = os.path.join(eval_folder,'label.txt')

    if os.path.exists(label_file) is False:
        raise FileNotFoundError()

    with open(label_file) as f:
        lines = f.read().splitlines()
        ground_truth_dict = { line.split(' ')[0]:line.split(' ')[1:] for line in lines }


    for k, v in ground_truth_dict.items():
        img = cv2.imread(os.path.join(eval_folder,k))
        boxes = yolo.predict(img, obj_threshold=0.1, nms_threshold=0.3)
        #



def filter_prediction(boxes, sub_threshold = 0.5):

    boxes = sorted(boxes, key=lambda box: box.get_score(),reverse=True)

    predictions = [ boxes[0] ]

    for i in range(len(boxes)):
        box = boxes[i]
        if box.get_score() > sub_threshold
            predictions.append(box)

    return predictions

if __name__ == '__main__':
    main()