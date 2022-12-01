  # 5#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 17:10:32 2022
@author: lee
The dual input is concatenate before input to the network, suitable for PSF varying case
"""
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import tifffile as tiff
from tensorflow.python.ops import gen_image_ops
import argparse
# import tflearn
import makedata3D_V3_dual2 as Input
import makedata3D_test_dual as InputT
import random
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

## mode: TR:train ; VL:validation, with known ground truth ; TS: test , no ground truth
global mode

mode = 'TS'

data_dir = '/home/liyue/newdata1/'
model_path = '/home/liyue/newdata1/train/model_rl/new_single_used/'
train_output = '/home/liyue/newdata1/train/output_rl/'
test_output = '/home/liyue/newdata1/test/output_rl/'

train_iter_num = 120 * 500  # 392*50
test_iter_num = 40
train_batch_size = 4
test_batch_size = 1

# is_training=True
input_data_depth = 128  # 128
input_data_width = 128  # 239#128
input_data_height = 128  # 386#128
input_data_channel = 1

output_data_depth = input_data_depth
output_data_width = input_data_width
output_data_height = input_data_height
output_data_channel = 1
EPS = 10e-5

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(train_output):
    os.makedirs(train_output)

if not os.path.exists(test_output):
    os.makedirs(test_output)


# 自定义两个函数
@tf.custom_gradient
def tensor_div(a, x1):
    m = tf.div_no_nan(a, x1 + 0.0001)  # 0.9*x1+0.1*a)

    def grad(dy):
        return dy, -tf.square(m) * dy

    return m, grad


@tf.custom_gradient
def tensor_mul(a, x):
    m = tf.multiply(a, x)

    def grad(dy):
        return dy, dy * (a)

    return m, grad


def global_avg_pool(x):
    b, _, _, _, c = x.shape
    return tf.reshape(tf.reduce_mean(x, axis=[1, 2, 3]), (b, 1, c))  # batch_size,1,channel


def chan_ave(x):
    weights = tf.reduce_mean(x, axis=[0, 1, 2, 3], keep_dims=True)
    print(weights)
    #    weights=weights/tf.reduce_sum(weights,axis=0)
    # print(weights)
    num = weights.shape[4]
    weights = tf.layers.dense(inputs=weights, units=num, activation=None)
    xo = weights * x
    #    xo=tf.reduce_sum(xo,axis=4,keep_dims=True)
    return xo


def s_sigmoid(x):
    return tf.nn.softplus(x)


def s_sigmoid1(x):
    return tf.nn.softplus(x)


class Unet:
    def __init__(self):
        self.input_image = {}
        self.ground_truth = {}
        self.cast_image = None
        self.cast_ground_truth = None
        self.is_traing = None
        self.m = None
        self.loss, self.loss_square, self.loss_all, self.train_step = [None] * 4
        self.prediction, self.correct_prediction, self.accuracy = [None] * 3
        self.result_conv = {}
        self.result_relu = {}
        self.result_from_contract_layer = {}
        self.w = {}
        self.a, self.b = [None] * 2
        self.sub_diff, self.mse_square, self.mse = [None] * 3
        self.mean_prediction, self.mean_gt = [None] * 2
        self.sigma_prediction, self.sigma_gt, self.sigma_cross = [None] * 3
        self.SSIM_1, self.SSIM_2, self.SSIM_3, self.SSIM_4, self.SSIM = [None] * 5
        self.learning_rate = [None]
        self.prediction_min = [None]

    def tf_fspecial_gauss(self, size, sigma1):
        x_data, y_data, z_data = np.mgrid[-size[0] // 2 + 1:size[0] // 2 + 1, -size[1] // 2 + 1:size[1] // 2 + 1,
                                 -size[2] // 2 + 1:size[2] // 2 + 1]
        #        print(x_data.shape)
        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        z_data = np.expand_dims(z_data, axis=-1)
        z_data = np.expand_dims(z_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)
        z = tf.constant(z_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma1 ** 2)))  # * tf.exp(-((z**2)/(2.0*9*sigma1**2)))
        g = g / tf.reduce_max(g)
        return g  # g/tf.reduce_max(g) #last_sum

    def init_w(self, shape, name, stddev=1.0):  # 0.5和1都可以
        with tf.name_scope('init_w'):
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, mean=0.0, stddev=stddev, dtype=tf.float32),
                            name=name)
            return w

    def gaussian_ker(self, shape, name, stddev=1):  # 1
        kernal2 = self.tf_fspecial_gauss(shape, stddev * 0.5)
        kernal3 = self.tf_fspecial_gauss(shape, stddev)
        kernal4 = self.tf_fspecial_gauss(shape, stddev * 1.5)
        kernal1 = self.tf_fspecial_gauss(shape, stddev * 2)
        rad = tf.random_uniform([1, 1, 1, shape[3], 1], minval=0, maxval=1, dtype=tf.float32)
        init = tf.concat([kernal1, kernal2, kernal3, kernal4], -1)
        init = tf.tile(init, [1, 1, 1, shape[3], 1]) * rad
        w = tf.Variable(initial_value=init, name=name)
        return w

    def gaussian_ker1(self, shape, name, stddev=1):  # 1
        kernal1 = self.gaussian_ker_single(shape, stddev)
        for i in range(shape[3] - 1):
            kernal = self.gaussian_ker_single(shape, stddev)
            kernal1 = tf.concat([kernal1, kernal], -2)
        print(kernal1.shape)
        w = tf.Variable(initial_value=kernal1, name=name)
        return w

    def srelu(self, x):
        a1 = tf.Variable(tf.constant(1.0), name='alpha', trainable=True)
        b1 = tf.Variable(tf.constant(1.0), name='beta', trainable=True)
        return tf.nn.relu(x / a1 + b1)

    def mul_attention(self, inputs, index, inner_units_ratio=1):
        with tf.variable_scope("CAB%s" % (index)):
            inputs_shape = map(lambda x: x.value, inputs.shape)
            b, d, h, w, c = inputs_shape
            GAP = global_avg_pool(inputs)
            fc_1 = tf.layers.dense(inputs=GAP, units=c * inner_units_ratio, name="fc_1", activation=None)
            fc_1 = tf.nn.relu(fc_1)
            fc_2 = tf.layers.dense(inputs=fc_1, units=c, name="fc_2", activation=None)
            fc_2 = tf.nn.sigmoid(fc_2)
            channel_attention = tf.reshape(fc_2, shape=[b, 1, 1, 1, c])
            out = tf.multiply(inputs, channel_attention)

            return out

    def leaky_relu(self, x, name='leaky_relu'):
        #       al=tf.Variable(tf.constant(0.03), name='beta', trainable=True)
        a = tf.nn.softplus(x)
        b = tf.nn.sigmoid(x / 2) * 2  ##放宽3倍没有2倍好,1也没有2好
        # m=tf.where(x>0,x=a,y=b)
        return s_sigmoid(x)  # 0.03

    ############### batch normalization  ###################
    @staticmethod
    def batch_norm(x, is_training, eps=EPS, name='BatchNorm3d'):  # GroupNorm(x,G=3,eps=1e-5):
        return tf.layers.batch_normalization(x, training=is_training)

    @staticmethod
    def copy_and_crop_and_merge(result_from_downsampling, result_from_upsampling):
        return tf.concat(values=[result_from_downsampling, result_from_upsampling], axis=-1)  ##axis=4

    def resize3D(self, x, shape):
        N, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
        N = tf.cast(N, tf.int32)
        D = tf.cast(D, tf.int32)
        H = tf.cast(H, tf.int32)
        W = tf.cast(W, tf.int32)
        C = tf.cast(C, tf.int32)
        c1 = tf.zeros([N, 1, H, W, C])
        c2 = tf.zeros([N, D + 1, 1, W, C])
        c3 = tf.zeros([N, D + 1, H + 1, 1, C])
        x1 = tf.concat([c1, x], axis=1)
        x2 = tf.concat([x, c1], axis=1)
        x_out = x1 + x2
        x1 = tf.concat([c2, x_out], axis=2)
        x2 = tf.concat([x_out, c2], axis=2)
        x_out = x1 + x2
        x1 = tf.concat([c3, x_out], axis=3)
        x2 = tf.concat([x_out, c3], axis=3)
        x_out = x1 + x2
        return x_out[:, :D, :H, :W, :]

    def get_SSIM(self, gt_label, dl_op, max_val=1):
        mean_prediction = tf.reduce_mean(dl_op)
        mean_gt = tf.reduce_mean(gt_label)
        sigma_prediction = tf.reduce_mean(tf.square(tf.subtract(dl_op, mean_prediction)))
        sigma_gt = tf.reduce_mean(tf.square(tf.subtract(gt_label, mean_gt)))
        sigma_cross = tf.reduce_mean(tf.multiply(tf.subtract(dl_op, mean_prediction),
                                                 tf.subtract(gt_label, mean_gt)))
        SSIM_1 = 2 * tf.multiply(mean_prediction, mean_gt) + 1e-4 * max_val * max_val
        SSIM_2 = 2 * sigma_cross + 9e-4 ** max_val * max_val
        SSIM_3 = tf.square(mean_prediction) + tf.square(mean_gt) + 1e-4 ** max_val * max_val
        SSIM_4 = sigma_prediction + sigma_gt + 9e-4 ** max_val * max_val
        SSIM = tf.div(tf.multiply(SSIM_1, SSIM_2), tf.multiply(SSIM_3, SSIM_4))
        return SSIM

    def get_mse(self, gt_label, dl_op):
        sub_diff = dl_op - gt_label
        mse_square = tf.square(sub_diff)
        MSE = tf.reduce_mean(mse_square) + 0.0001
        return MSE

    def get_mae(self, gt_label, dl_op):
        sub_diff = tf.abs(dl_op - gt_label)
        MAE = tf.reduce_mean(sub_diff)
        return MAE

    def set_up_unet(self, batch_size):
        # input
        with tf.name_scope('input'):
            self.shape = tf.placeholder(dtype=tf.int32)
            self.input_image1 = tf.placeholder(dtype=tf.float32)
            self.input_image2 = tf.placeholder(dtype=tf.float32)
            self.ground_truth = tf.placeholder(dtype=tf.float32)
            self.cast_image1 = tf.reshape(
                tensor=self.input_image1,
                shape=[batch_size, self.shape[0], self.shape[1], self.shape[2], input_data_channel]
            )

            self.cast_image2 = tf.reshape(
                tensor=self.input_image2,
                shape=[batch_size, self.shape[0], self.shape[1], self.shape[2], input_data_channel]
            )

            self.cast_ground_truth = tf.reshape(
                tensor=self.ground_truth,
                shape=[batch_size, self.shape[0], self.shape[1], self.shape[2], output_data_channel]
            )


            self.is_traing = tf.placeholder(tf.bool)


        with tf.name_scope('input'):
            self.cast_ground_truth1 = self.cast_ground_truth * 0.8 + 0.2 * (self.cast_image1 + self.cast_image2) / 2
            normed_batch=tf.concat([self.cast_image1,self.cast_image2],-1)
            normed_batch1=(self.cast_image1+self.cast_image2)/2

        # layer 1
        with tf.name_scope('estimation'):
            # conv_1
            m = tf.reduce_max(normed_batch)
            normed_batch_t_down = tf.nn.avg_pool3d(normed_batch, [1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                                                   padding='VALID', data_format='NDHWC')
            normed_batch_t_down_m = s_sigmoid(tf.tile(input=normed_batch_t_down, multiples=[1, 1, 1, 1, 2]))
            self.w[3] = self.gaussian_ker(shape=[3, 3, 3, 2, 4], name='e_1')  # 9,9,9
            result_conv_3 = tf.nn.conv3d(
                input=normed_batch_t_down, filter=self.w[3],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
            result_prelu_3 = s_sigmoid1(self.batch_norm(x=result_conv_3, is_training=self.is_traing, name='eb_1'))

            self.w[9] = self.gaussian_ker(shape=[3, 3, 3, 4, 4], name='e_2')
            result_conv_9 = tf.nn.conv3d(
                input=result_prelu_3, filter=self.w[9],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_9 = s_sigmoid1(
                self.batch_norm(x=result_conv_9, is_training=self.is_traing, name='eb_2'))  # +normed_batch_t_down_m

            result_conv_9_1 = tf.concat([result_prelu_3, result_conv_9], -1)

            self.w[91] = self.gaussian_ker(shape=[3, 3, 3, 8, 4], name='e_3')
            result_conv_91 = tf.nn.conv3d(
                input=result_conv_9_1, filter=self.w[91],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_3')  # +normed_batch_t_down_m
            normed_batch_9 = self.batch_norm(x=result_conv_91, is_training=self.is_traing, name='eb_3')
            result_prelu_9 = s_sigmoid(normed_batch_9) + normed_batch_t_down_m

            ave_9 = tf.reduce_mean(result_prelu_9, axis=4, keep_dims=True)
            ave_9=tf.concat([ave_9,ave_9],-1)

            temp_layer = tensor_div(normed_batch_t_down, ave_9)
            temp_layer = self.batch_norm(x=temp_layer, is_training=self.is_traing, name='t_1')

            self.w[15] = self.init_w(shape=[3, 3, 3, 2, 8], name='e_4')
            result_conv_15 = tf.nn.conv3d(
                input=temp_layer, filter=self.w[15],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_4')
            result_prelu_15 = s_sigmoid1(self.batch_norm(x=result_conv_15, is_training=self.is_traing, name='eb_4'))

            self.w[10] = self.init_w(shape=[3, 3, 3, 8, 8], name='e_5')  # 18,1,1
            result_conv_10 = tf.nn.conv3d(
                input=result_prelu_15, filter=self.w[10],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_5')
            result_prelu_10 = s_sigmoid1(self.batch_norm(x=result_conv_10, is_training=self.is_traing, name='eb_5'))

            result_prelu_10_1 = tf.concat([result_prelu_10, result_prelu_15], -1)

            self.w[12] = self.init_w(shape=[3, 3, 3, 16, 8], name='e_6')
            result_conv_12 = tf.nn.conv3d(
                input=result_prelu_10_1, filter=self.w[12],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_6')
            normed_batch_12 = self.batch_norm(x=result_conv_12, is_training=self.is_traing, name='eb_6')
            result_prelu_12 = s_sigmoid1(normed_batch_12)

            self.w[20] = self.init_w(shape=[2, 2, 2, 4, 8], name='e_7')
            result_conv_12_u = tf.nn.conv3d_transpose(
                value=result_prelu_12, filter=self.w[20],
                output_shape=[batch_size, self.shape[0], self.shape[1], self.shape[2], 4],
                strides=[1, 2, 2, 2, 1], padding='VALID', name='conv_7')
            result_conv_12_u = s_sigmoid1(self.batch_norm(x=result_conv_12_u, is_training=self.is_traing, name='eb_7'))
            self.w[205] = self.init_w(shape=[3, 3, 3, 4, 4], name='e_8')
            result_conv_12_u2 = tf.nn.conv3d(
                input=result_conv_12_u, filter=self.w[205],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_8')
            normed_batch_12_u2 = self.batch_norm(x=result_conv_12_u2, is_training=self.is_traing, name='eb_8')
            result_prelu_12_u2 = s_sigmoid(normed_batch_12_u2)

            ave_result_prelu_12_u2 = tf.reduce_mean(result_prelu_12_u2, axis=4, keep_dims=True)
            temp2 = tensor_mul(normed_batch1, ave_result_prelu_12_u2)
            result_prelu_12_1 = s_sigmoid(self.batch_norm(x=temp2, is_training=self.is_traing, name='t_2'))

        with tf.name_scope('update'):
            normed_batch_m = s_sigmoid(tf.tile(input=normed_batch, multiples=[1, 1, 1, 1, 2]))
            self.w[1] = self.gaussian_ker(shape=[3, 3, 3, 2, 4], name='u_1')  # +self.w[3]
            result_conv_1 = tf.nn.conv3d(
                input=normed_batch, filter=self.w[1],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_9')
            result_conv_1_b = s_sigmoid1(self.batch_norm(x=result_conv_1, is_training=self.is_traing, name='ub_1'))

            self.w[101] = self.gaussian_ker(shape=[3, 3, 3, 4, 4], name='u_2')
            result_conv_1_1 = tf.nn.conv3d(
                input=result_conv_1_b, filter=self.w[101],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_10')
            normed_batch_1 = self.batch_norm(x=result_conv_1_1, is_training=self.is_traing, name='ub_2')
            result_prelu_1 = s_sigmoid(normed_batch_1) + normed_batch_m

            ave_1 = tf.reduce_mean(result_prelu_1, axis=4, keep_dims=True)
            ave_1=tf.concat([ave_1,ave_1],-1)
            EST = tensor_div(normed_batch, ave_1)
            EST = self.batch_norm(x=EST, is_training=self.is_traing, name='t_3')

            self.w[2] = self.init_w(shape=[3, 3, 3, 2, 8], name='u_3')
            result_conv_2 = tf.nn.conv3d(
                input=EST, filter=self.w[2],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_11')
            result_conv_2_b = s_sigmoid1(self.batch_norm(x=result_conv_2, is_training=self.is_traing, name='ub_3'))

            self.w[201] = self.init_w(shape=[3, 3, 3, 8, 8], name='u_4')
            result_conv_2_1 = tf.nn.conv3d(
                input=result_conv_2_b, filter=self.w[201],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_12')
            normed_batch_2 = self.batch_norm(x=result_conv_2_1, is_training=self.is_traing, name='u_4')
            act_2 = s_sigmoid(normed_batch_2) + tf.ones_like(normed_batch_2)

            ave_2 = tf.reduce_mean(act_2, axis=4, keep_dims=True)

            Estimation1 = tensor_mul(result_prelu_12_1, ave_2)
            Estimation = s_sigmoid(self.batch_norm(x=Estimation1, is_training=self.is_traing, name='t_4'))
            result_prelu_2 = Estimation
            result_prelu_2_1 = Estimation

            Estimation_tile = tf.tile(input=Estimation, multiples=[1, 1, 1, 1, 8])
            self.w[202] = self.init_w(shape=[3, 3, 3, 1, 8], name='u_5')
            result_conv_2_fine1 = tf.nn.conv3d(
                input=result_prelu_2_1, filter=self.w[202],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_13')
            # normed_batch_2_fine1 = self.batch_norm(x=result_conv_2_fine1, is_training=self.is_traing, name='n_3_1')
            result_conv_2_fined = s_sigmoid1(result_conv_2_fine1)

            result_conv_2_fine1_c = tf.concat([result_conv_2_fined, result_prelu_2_1, result_prelu_12_1],
                                              -1)  # if not ok, try result_prelu_12_1

            self.w[203] = self.init_w(shape=[3, 3, 3, 10, 8], name='u_6')
            result_conv_2_fine = tf.nn.conv3d(
                input=result_conv_2_fine1_c, filter=self.w[203],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_14')
            act_2_fine = s_sigmoid1(result_conv_2_fine)

            Merge = tf.concat([result_conv_2_fined, act_2_fine], -1)

            self.w[13] = self.init_w(shape=[3, 3, 3, 16, 8], name='u_7')
            result_conv_13 = tf.nn.conv3d(
                input=Merge, filter=self.w[13],
                strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_15')
            normed_batch_13 = self.batch_norm(x=result_conv_13, is_training=self.is_traing, name='ub_7')
            result_prelu_13 = s_sigmoid(normed_batch_13)

            self.prediction = tf.reduce_mean(result_prelu_13, axis=4, keep_dims=True)
            self.prediction_log = self.prediction
            #            print(self.prediction.shape)

            self.e3 = normed_batch
            self.e = temp2
            self.e2 = Estimation1

            self.first = tf.reduce_mean(result_prelu_13, axis=4, keep_dims=True)
            #    self.prediction_nor=self.prediction/tf.reduce_max(self.prediction)#tf.cond(self.prediction_min<0,lambda: self.prediction/tf.reduce_max(self.prediction), lambda:(self.prediction-self.prediction_min)/(tf.reduce_max(self.prediction)-self.prediction_min))

            # tvars = tf.trainable_variables()
            # e_vars = [var for var in tvars if 'e_' in var.name]
            e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estimation')
            w_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='update')

        with tf.name_scope('MSE'):
            self.mse = 0.0 * self.get_mae(self.prediction_log, self.cast_ground_truth) + 1.0 * self.get_mse(
                self.prediction_log, self.cast_ground_truth)

        with tf.name_scope('SSIM'):
            self.SSIM = self.get_SSIM(self.prediction_log, self.cast_ground_truth)

        with tf.name_scope('SSIM2'):
            self.SSIM2 = self.get_mse(self.e, self.cast_ground_truth1)
            self.mse2 = self.SSIM2  # -tf.log((1+self.SSIM2)/2)#+1.0*self.SSIM1#tf.log((1+self.SSIM1)/2)

        with tf.name_scope('loss'):
            k1 = 0.0  # 1.0 #1.0
            self.loss = 0.1 * self.mse2 + 1 * self.mse - 1.0 * tf.log((1 + self.SSIM) / 2)  # -k1*self.prediction_min

        # Gradient Descent
        with tf.name_scope('step'):
            self.global_step = tf.Variable(0, trainable=False)  # 0.015,500,0.95
            self.learning_rate = tf.train.exponential_decay(0.02, self.global_step, 500, 0.95,
                                                            staircase=False)  # previous0.025,240,0.9
        # with tf.name_scope('Gradient_Descent'):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                global_step=self.global_step)

    # easy
    # 12data:0.02,200,0.9

    def train(self):
        Input.filenames = []
        train_dir = model_path
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        pre_parameters_saver = tf.train.Saver()  # sav
        all_parameters_saver = tf.train.Saver()  # save
        # tf.reset_default_graph()
        with tf.Session(config=config) as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            #    all_parameters_saver.restore(sess=sess, save_path=checkpoint_path)
            sum_los = 0.0
            time_start = time.time()
            for k in range(train_iter_num):
                # t1=time.time()
                train_images1, train_images2, train_GT, label2out, shape = Input.get_data_tiff(data_dir, 'train',
                                                                                               train_batch_size, True)
                # t2=time.time()
                # print(t2-t1)
                mse, ssim, lo, mse2, trainer = sess.run([self.mse, self.SSIM, self.loss, self.mse2, self.train_step],
                                                        feed_dict={self.shape: shape, self.input_image1: train_images1,
                                                                   self.input_image2: train_images2,
                                                                   self.ground_truth: train_GT, self.is_traing: True})
                # t3=time.time()
                # print(t3-t2)
                sum_los += lo
                if k % 101 == 0:
                    time_end = time.time()
                    used_time = time_end - time_start
                    print('num %d, mse: %.6f, SSIM: %.6f, loss: %.6f,mse2: %.6f. runtime:%.6f ' % (
                    k, mse, ssim, lo, mse2, used_time))
                if (k + 1) % 120 == 0:  # 13
                    print('sum_lo: %.6f' % (sum_los))
                    sum_los = 0.0
                if (k + 1) % 100000 == 0:
                    image = sess.run([self.prediction],
                                     feed_dict={self.shape: shape, self.input_image1: train_images1,
                                                self.input_image2: train_images2, self.ground_truth: train_GT,
                                                self.is_traing: True})
                    image1 = np.array(image)
                    print(image1.shape)
                    reshape_image = image1[0, :, :, :, :, 0]
                    print(label2out)
                    #                    print('sum_lo: %.6f' %(sum_los))
                    #                    sum_los = 0.0
                    for v in range(train_batch_size):
                        single = reshape_image[v]
                        filenames_out = train_output + str(k) + '_' + label2out[v]
                        tiff.imsave(filenames_out, single)
                if (k + 1) % 100 == 0:
                    all_parameters_saver.save(sess=sess, save_path=checkpoint_path)
            print("Done training")
        sess.close()

    def val(self):
        Input.filenames = []
        train_dir = model_path
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            #            time_start=time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            all_parameters_saver.restore(sess=sess, save_path=checkpoint_path)
            sum_los = 0.0
            for m in range(test_iter_num):
                time_start = time.time()
                test_images, test_GT, label2out, shape = Input.get_data_tiff_VL(data_dir, 'test', test_batch_size)
                w1, image, mse, ssim, los = sess.run([self.w[1], self.prediction, self.mse, self.SSIM, self.loss],
                                                     feed_dict={self.shape: shape, self.input_image1: test_images1,
                                                                self.input_image2: test_images2,
                                                                self.ground_truth: test_GT, self.is_traing: False})
                sum_los += los
                image1 = (np.array(image))
                print(image1.shape)
                reshape_image = image1[:, :, :, :, 0]
                for v in range(test_batch_size):
                    single = reshape_image[v]
                    filenames_out = test_output + 'rl_' + label2out[v]
                    tiff.imsave(filenames_out, single)
                #########  save output image #####################
                if m % 1 == 0:
                    time_end = time.time()
                    print('num %d, mse: %.6f, SSIM: %.6f, loss: %.6f, runtime:%.6f ' % (
                    m, mse, ssim, los, time_end - time_start))
        print('Done validating')

    def test1(self):
        Input.filenames = []
        train_dir = model_path
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            #            time_start=time.time()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            all_parameters_saver.restore(sess=sess, save_path=checkpoint_path)
            for m in range(test_iter_num):
                time_start = time.time()
                test_images1, test_images2, label2out, shape = InputT.get_data_tiff(data_dir, 'test', test_batch_size)
                image = sess.run([self.first], feed_dict={self.shape: shape, self.input_image1: test_images1,
                                                          self.input_image2: test_images2, self.is_traing: False})
                image1 = (np.array(image))
                reshape_image = image1[:, :, :, :]
                for v in range(test_batch_size):
                    single = reshape_image[v]
                    filenames_out = test_output + 'rl_' + label2out[v]
                    tiff.imsave(filenames_out, single)
                #########  save output image #####################
                if m % 1 == 0:
                    time_end = time.time()
                    print('num %d, runtime:%.6f ' % (m, time_end - time_start))
        print('Done testing')


def main():
    net = Unet()
    if mode == 'TR':
        net.set_up_unet(train_batch_size)
        net.train()
    if mode == 'VL':
        net.set_up_unet(VALIDATION_BATCH_SIZE)
        net.val()
    if mode == 'TS':
        net.set_up_unet(test_batch_size)
        net.test1()


main()