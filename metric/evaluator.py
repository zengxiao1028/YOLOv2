import numpy as np
import os
import cv2
from utils import BoundBox, bbox_iou
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
def gen_ground_truth_boxes(gt_sample_objects,labels):
    gt_boxes = []

    for object in gt_sample_objects:
        x = (object['xmin'] + object['xmax'])/2
        y = (object['ymin'] + object['ymax'])/2
        w = object['xmax'] - object['xmin']
        h = object['ymax'] - object['ymin']
        bb = BoundBox(x, y, w, h)
        bb.label = labels.index(object['name'])
        gt_boxes.append(bb)

    assert(len(gt_boxes)>0)

    return gt_boxes

def evaluate_img(eval_sample, yolo, config, result_dict, iou_threshold):

    img = cv2.imread(eval_sample['filename'])

    # obtain predicted boxes
    predicted_boxes = yolo.predict( img ,config["valid"]["obj_threshold"],config["valid"]["nms_threshold"])
    for box in predicted_boxes:
        box.x *= img.shape[1]
        box.w *= img.shape[1]
        box.y *= img.shape[0]
        box.h *= img.shape[0]

    # obtain gt boxes
    ground_truth_boxes = gen_ground_truth_boxes(eval_sample['object'], config['model']['labels'])
    boxes_det          = np.zeros( (len(ground_truth_boxes),),dtype=bool)


    for pred_box in predicted_boxes:


        # find the best matched gt_box for this pred_box
        iou_max = -100000.
        fp = 0
        tp = 0
        for i, gt_box in enumerate(ground_truth_boxes):

            # class does not match, skip it.
            if pred_box.get_label() == gt_box.get_label():

                iou = bbox_iou(pred_box,gt_box)

                if iou>iou_max:
                    iou_max = iou
                    i_max = i  # which gt bbox is going to be assigned

        # if a gt box with corrected classs is found and its iou is greated than threshold
        if iou_max >= iou_threshold:

            if boxes_det[i_max] == True:
                # if this gt box is already assigned to another predicted box
                fp = 1
            else:
                tp = 1
                boxes_det[i_max] = True

        # otherwise it is a false negative
        else:
            fp = 1

        result_dict[pred_box.get_label()].append((pred_box, tp, fp, eval_sample['filename']))

def evaluate(eval_samples, yolo, config, iou_threshold=0.5):


    predictions_dict = defaultdict(list)
    nb_pos_dict = defaultdict(int)
    result_dict = defaultdict(dict)

    for sample in eval_samples:
        evaluate_img(sample, yolo, config, predictions_dict, iou_threshold)
        for obj in sample['object']:
            label = config['model']['labels'].index(obj['name'])
            nb_pos_dict[label] += 1

    # compute ap for each class
    for label in predictions_dict.keys():

        # use predict score as key to sort
        predictions_dict[label].sort(reverse=True, key=lambda tup: tup[0].get_score())

        # (pred_box, tp, fp, eval_sample['filename'])
        cumsum_tp = np.cumsum( [prediction[1] for prediction in predictions_dict[label]] )
        cumsum_fp = np.cumsum( [prediction[2] for prediction in predictions_dict[label]] )

        recall = cumsum_tp * 1.0 / nb_pos_dict[label]
        precision = cumsum_tp * 1.0 / (cumsum_tp + cumsum_fp)

        result_dict[label]['recall']=recall
        result_dict[label]['precision']= precision

        ap = 0
        for x in range(0, 11):
            t = x/10.
            tmp = precision[recall >= t]
            if len(tmp)==0:
                p = 0
            else:
                p = max( tmp )
            ap += p / 11.
        result_dict[label]['ap'] = ap


    #compute mAP
    mAP = 0
    nb_classes = 0
    for label in result_dict.keys():
        mAP += result_dict[label]['ap']
        nb_classes += 1
    mAP = mAP / nb_classes

    print('mAP', mAP)
    return result_dict, mAP


def sumnmarize_result(result, labels, save_folder ='/tmp'):

    result_dict, mAP = result
    marker = ('d', 'h', '*', '<', 'o','s','v','^','p')
    my_color = ('gold', 'red', 'green', 'magenta', 'peru','darkgray','indigo','blue','lime')
    markers = itertools.product(marker,my_color)

    os.makedirs(save_folder,exist_ok=True)
    ### plot pr curve for each class
    for label_idx in result_dict.keys():
        marker,my_color =  next(markers)
        #precision-recall curve
        xs = result_dict[label_idx]['recall']
        ys = result_dict[label_idx]['precision']

        plt.plot(xs,ys,label=labels[label_idx],marker=marker, color=my_color, linewidth=1,  markersize=5)


    plt.title('Precision-Recall Curves, mAP: %.2f',result_dict['mAP'])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(labels,loc='center left', bbox_to_anchor=(1,0.5))
    plt.savefig(os.path.join(save_folder, 'pr_curve.pdf'), bbox_inches='tight')
    plt.show()
    plt.close()







if __name__ == '__main__':

    evaluate()
