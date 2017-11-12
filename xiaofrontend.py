from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
import os
import tensorflow as tf
import numpy as np
import cv2
from keras.applications.mobilenet import MobileNet
import shutil
from keras.layers.advanced_activations import LeakyReLU
from preprocessing import BatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from utils import BoundBox
from backend import TinyYoloFeature, FullYoloFeature, MobileNetFeature, SqueezeNetFeature, Inception3Feature
from keras.utils import multi_gpu_model
from keras import regularizers
from frontend import YOLO
from utils import space_to_depth_x2
class XiaoYOLO(YOLO):


    def __init__(self, architecture,
                 input_size,
                 labels,
                 max_box_per_image,
                 anchors,
                 obj_threshold=0.3,
                 weight_decay=None):

        self.input_size = input_size
        self.obj_threshold = obj_threshold
        self.labels = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box = max_box_per_image  # num of boxes for each cell
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors = anchors

        self.max_box_per_image = max_box_per_image

        #### weight_decay
        kr = None if weight_decay is None or weight_decay == 0 else regularizers.l2(weight_decay)

        ##########################
        # Make the model
        ##########################
        # make the feature extractor layers
        input_image = Input(shape=(self.input_size, self.input_size, 3))
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

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
            raise Exception(
                'Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, and Inception3 at the moment!')

        print(self.feature_extractor.get_output_shape())
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()

        #which level of feature to use
        features = self.feature_extractor.extract(input_image)

        low_features = self.feature_extractor.feature_extractor_model.get_layer('out_2').output

        output = self._make_head(low_features)


        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)

        # initialize the weights of the detection layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()
        new_kernel = np.random.normal(size=weights[0].shape) / (self.grid_h * self.grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self.grid_h * self.grid_w)
        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self.model.summary()

    def _make_low_head(self, features, kr):

        # sub_conv1
        sub_x = Conv2D(32,
                       (3, 3), strides=(1, 1),
                       dilation_rate=2,
                       padding='same',
                       name='sub_conv1',
                       kernel_regularizer=kr)(features)
        sub_x = BatchNormalization(name='sub_norm1')(sub_x)
        sub_x = LeakyReLU(alpha=0.1)(sub_x)

        # sub_conv2
        sub_x = Conv2D(32,
                       (3, 3), strides=(1, 1),
                       dilation_rate=4,
                       padding='same',
                       name='sub_conv2',
                       kernel_regularizer=kr)(sub_x)
        sub_x = BatchNormalization(name='sub_norm2')(sub_x)
        sub_x = LeakyReLU(alpha=0.1)(sub_x)

        # sub_conv3
        sub_x = Conv2D(64,
                       (3, 3), strides=(1, 1),
                       dilation_rate=8,
                       padding='same',
                       name='sub_conv3',
                       kernel_regularizer=kr)(sub_x)
        sub_x = BatchNormalization(name='sub_norm3')(sub_x)
        sub_x = LeakyReLU(alpha=0.1)(sub_x)

        # sub_conv4
        sub_x = Conv2D(128,
                       (3, 3), strides=(1, 1),
                       dilation_rate=16,
                       padding='same',
                       name='sub_conv4',
                       kernel_regularizer=kr)(sub_x)
        sub_x = BatchNormalization(name='sub_norm1')(sub_x)
        sub_x = LeakyReLU(alpha=0.1)(sub_x)

        # resize to target
        sub_x = Lambda(lambda x: tf.space_to_depth(x, block_size=8))(sub_x)

        sub_x = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                            (1, 1), strides=(1, 1),
                            padding='same',
                            name='sub_conv5',
                            kernel_regularizer=kr)(sub_x)
        sub_output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(sub_x)

        return sub_output


    def _make_head(self, features, kr):

        # make the object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='conv_23',
                        kernel_initializer='lecun_normal', kernel_regularizer=kr)(features)
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)


        return output
