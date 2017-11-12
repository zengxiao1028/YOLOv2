from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
import os
import tensorflow as tf
import numpy as np
from keras.layers.merge import concatenate
from keras.applications.mobilenet import MobileNet
import cv2
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

        features = self._make_body(input_image, kr)

        print(features.shape)

        self.grid_h, self.grid_w = features.shape.as_list()[1:3]

        output = self._make_head(features, kr)

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

    def _make_body(self, input_image, kernel_regularizer):
        # Layer 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), name='out_1')(x)

        # Layer 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), name='out_2')(x)

        # Layer 3
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1, name='out_3')(x)

        # Layer 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1, name='out_4')(x)

        # Layer 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), name='out_5')(x)

        # Layer 6
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1, name='out_6')(x)

        # Layer 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1, name='out_7')(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), name='out_8')(x)

        # Layer 9
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1, name='out_9')(x)

        # Layer 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1, name='out_10')(x)

        # Layer 11
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1, name='out_11')(x)

        # Layer 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1, name='out_12')(x)

        # Layer 13
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1, name='out_13')(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1, name='out_14')(x)

        # Layer 15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1, name='out_15')(x)

        # Layer 16
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1, name='out_16')(x)

        # Layer 17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1, name='out_17')(x)

        # Layer 18
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1, name='out_18')(x)

        # Layer 19
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1, name='out_19')(x)

        # Layer 20
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1, name='out_20')(x)

        # Layer 21
        skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False,
                                 kernel_regularizer=kernel_regularizer)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(lambda y: tf.space_to_depth(y, block_size=2))(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False,
                   kernel_regularizer=kernel_regularizer)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1, name='out_22')(x)
        return x

    def _make_sub_head(self, features, kr):

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
        sub_x = BatchNormalization(name='sub_norm4')(sub_x)
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



    def normalize(self,image):
        return image / 255.
