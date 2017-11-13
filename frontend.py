from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
import os
import tensorflow as tf
import numpy as np
import cv2
from keras.applications.mobilenet import MobileNet
import shutil
from keras.optimizers import SGD, Adam, RMSprop
from preprocessing import BatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from utils import BoundBox
from backend import TinyYoloFeature, FullYoloFeature, MobileNetFeature, SqueezeNetFeature, Inception3Feature
from keras.utils import multi_gpu_model
from keras import regularizers
class YOLO(object):


    def __init__(self, architecture,
                       input_size, 
                       labels, 
                       max_box_per_image,
                       anchors,
                       obj_threshold=0.3,
                       weight_decay = None):

        self.input_size = input_size
        self.obj_threshold = obj_threshold
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box   = max_box_per_image       #num of boxes for each cell
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors  = anchors

        self.max_box_per_image = max_box_per_image


        #### weight_decay
        kr = None if weight_decay is None or weight_decay==0 else regularizers.l2(weight_decay)

        ##########################
        # Make the model
        ##########################
        # make the feature extractor layers
        input_image     = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))  

        if architecture == 'Inception3':
            self.feature_extractor = Inception3Feature(self.input_size)
        elif architecture == 'SqueezeNet':
            self.feature_extractor = SqueezeNetFeature(self.input_size)
        elif architecture == 'MobileNet':
            self.feature_extractor = MobileNetFeature(self.input_size)
        elif architecture == 'Full Yolo':
            self.feature_extractor = FullYoloFeature(self.input_size, kernel_regularizer=kr)
        elif architecture == 'Tiny Yolo':
            self.feature_extractor = TinyYoloFeature(self.input_size)
        else:
            raise Exception('Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, and Inception3 at the moment!')

        print(self.feature_extractor.get_output_shape())
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()
        features = self.feature_extractor.extract(input_image)

        # make the object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class), 
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='conv_23', 
                        kernel_initializer='lecun_normal',kernel_regularizer=kr)(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)
        
        # initialize the weights of the detection layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)

        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self.model.summary()

    @classmethod
    def init_from_config(cls, config):

        yolo = cls(architecture=config['model']['architecture'],
                   input_size=config['model']['input_size'],
                   labels=config['model']['labels'],
                   max_box_per_image=config['model']['max_box_per_image'],
                   anchors=config['model']['anchors'],
                   obj_threshold=config["valid"]["obj_threshold"],
                   weight_decay=config['train']['weight_decay'] if 'weight_decay' in config['train'].keys() else None)

        return yolo

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]
        
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, 5, 1])
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        
        total_recall = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.nb_box,2])
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., 4]
        
        ### adjust class probabilities
        true_box_class = tf.to_int32(y_true[..., 5])
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale       
        
        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale/2.)
        seen = tf.assign_add(seen, 1.)
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_bs), 
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                       true_box_wh + tf.ones_like(true_box_wh) * np.reshape(self.anchors, [1,1,1,self.nb_box,2]) * no_boxes_mask, 
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       coord_mask])
        
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = loss_xy + loss_wh + loss_conf + loss_class
        
        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > self.obj_threshold))
            
            current_recall = nb_pred_box/(nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            loss = tf.Print(loss, [loss_xy, loss_wh, loss_conf, loss_class, loss, current_recall, total_recall / seen],
                            message='DEBUG', summarize=1000)
        return loss

    def load_YOLO_official_weights(self, weight_path):
        #load feature extractor model weights
        try:
            self.feature_extractor.feature_extractor_model.load_weights(weight_path, by_name=True)
        except ValueError as err:
            print("Error Loading Feature Extractor weights:",err)
        except AttributeError as err:
            print("Error Loading Feature Extractor weights:", err)
            pass

        # load feature extractor model weights
        try:
            self.model.load_weights(weight_path,by_name=True)
        except ValueError as err:
            print("Error Loading Top layer weights:",err)
            pass


    def load_weights(self, weight_path):

        # load feature extractor model weights

        self.model.load_weights(weight_path,by_name=True)



    def freeze_layers(self, to_layer):

        # layer_names = [layer.name for layer in self.feature_extractor.feature_extractor_model.layers]
        # for i,each in enumerate(layer_names):
        #     print(i,each)
        #
        for layer in self.feature_extractor.feature_extractor_model.layers[:to_layer]:
            layer.trainable = False

    def normalize(self,image):
        return self.feature_extractor.normalize(image)

    def predict(self, image,obj_threshold=0.3, nms_threshold=0.3):
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = self.decode_netout(netout, obj_threshold, nms_threshold)
        
        return boxes

    def bbox_iou(self, box1, box2):
        x1_min  = box1.x - box1.w/2
        x1_max  = box1.x + box1.w/2
        y1_min  = box1.y - box1.h/2
        y1_max  = box1.y + box1.h/2
        
        x2_min  = box2.x - box2.w/2
        x2_max  = box2.x + box2.w/2
        y2_min  = box2.y - box2.h/2
        y2_max  = box2.y + box2.h/2
        
        intersect_w = self.interval_overlap([x1_min, x1_max], [x2_min, x2_max])
        intersect_h = self.interval_overlap([y1_min, y1_max], [y2_min, y2_max])
        
        intersect = intersect_w * intersect_h
        
        union = box1.w * box1.h + box2.w * box2.h - intersect
        
        return float(intersect) / union
        
    def interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3          

    def decode_netout(self, netout, obj_threshold=0.3, nms_threshold=0.3):
        grid_h, grid_w, nb_box = netout.shape[:3]

        boxes = []
        
        # decode the output by the network
        netout[..., 4]  = self.sigmoid(netout[..., 4]) # squash confidence into [0, 1]
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * self.softmax(netout[..., 5:]) # confidence * prediction
        netout[..., 5:] *= netout[..., 5:] > obj_threshold # t
        
        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row,col,b,5:]
                    
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row,col,b,:4]

                        x = (col + self.sigmoid(x)) / grid_w # center position, unit: image width
                        y = (row + self.sigmoid(y)) / grid_h # center position, unit: image height
                        w = self.anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                        h = self.anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                        confidence = netout[row,col,b,4]
                        
                        box = BoundBox(x, y, w, h, confidence, classes)
                        
                        boxes.append(box)

        # suppress non-maximal boxes
        for c in range(self.nb_class):
            sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                
                if boxes[index_i].classes[c] == 0: 
                    continue
                else:
                    for j in range(i+1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        
                        if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                            boxes[index_j].classes[c] = 0
                            
        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.get_score() > obj_threshold]
        
        return boxes

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)
        
        if np.min(x) < t:
            x = x/np.min(x)*t
            
        e_x = np.exp(x)
        
        return e_x / e_x.sum(axis, keepdims=True)

    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    train_times,    # the number of time to repeat the training set, often used for small datasets
                    valid_times,    # the number of times to repeat the validation set, often used for small datasets
                    nb_epoch,       # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    warmup_bs,      # number of initial batches to let the model familiarize with the new dataset
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    config_path,
                    saved_weights_name='best_weights.h5',
                    name = 'YOLOv2',
                    debug=False,
                    nb_gpus=1):

        self.batch_size = batch_size
        self.warmup_bs  = warmup_bs 

        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale

        self.debug = debug

        ############################################
        # Compile the model
        ############################################

        if nb_gpus>1:
            print("Using %d GPUs For Training" % nb_gpus)
            parallel_model = multi_gpu_model(self.model, nb_gpus)
            parallel_model.callback_model = self.model
            self.model = parallel_model

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer,metrics=[self.custom_loss])

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_size, 
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,  
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }    

        train_batch = BatchGenerator(train_imgs,
                                     generator_config,
                                     norm=self.normalize)
        valid_batch = BatchGenerator(valid_imgs,
                                     generator_config,
                                     norm=self.normalize,
                                     jitter=False)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=10,
                           mode='min', 
                           verbose=1)

        training_save_dir = './traning_results/'
        os.makedirs(training_save_dir,exist_ok=True)
        result_counter = len([log for log in os.listdir(training_save_dir) if name == log.split('_')[:-1].join('_')]) + 1
        saved_dir = os.path.join(training_save_dir,name + '_' + str(result_counter))
        os.makedirs(saved_dir, exist_ok=True)
        shutil.copyfile(config_path,os.path.join(saved_dir,'config.json'))
        best_checkpoint = ModelCheckpoint(os.path.join(saved_dir,'best_' + saved_weights_name),
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
        checkpoint = ModelCheckpoint(os.path.join(saved_dir, saved_weights_name),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False,
                                     mode='min',
                                     period=1)

        tensorboard = TensorBoard(log_dir=saved_dir,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)


        ############################################
        # Start the training process
        ############################################
        self.model.fit_generator(generator        = train_batch,
                                 steps_per_epoch  = len(train_batch) * train_times,
                                 epochs           = nb_epoch,
                                 validation_data  = valid_batch,
                                 validation_steps = len(valid_batch) * valid_times,
                                 callbacks        = [best_checkpoint, checkpoint, tensorboard],
                                 workers = 4,
                                 max_queue_size   = 32)
