#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:57:35 2018

@author: lee
"""

# A script to load images and make batch.

import os
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import random
import tifffile as tiff
np.seterr(divide='ignore', invalid='ignore')

FLAGS = tf.app.flags.FLAGS
width = 128#211 #define width
height =128#326  #define height
depth = 128#191 #define depth
batch_index = 0
filenames = []

# user selection
#data_dir = '/home/liyue/newdata1/' #define data dictionary，include 'train','test',and include 'input' and 'ground truth' as subsictionary
#FLAGS.input_dir = '/home/ikbeom/Desktop/DL/MNIST_simpleCNN/data' #define input dictionary
#FLAGS.GT_dir = '/home/ikbeom/Desktop/DL/MNIST_simpleCNN/data'  #define ground truth dictionary

def get_filenames(data_dir,data_set): #获取文件名称
    global filenames
    labels = []
    
    with open(data_dir +data_set + '/labels.txt') as f: #labels.txt use to name the input
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')] #以逗号为分隔符，分开一行中的话，并去空格赋值入inner_list中,我的输入输出共享同样的文件名但所在文件夹不一样，每一行代表文件名？
            labels += inner_list  #在我的数据中没有label，只有input和ground_truth
            
        for i, label in enumerate(labels):
            #list = os.listdir(FLAGS.data_dir  + '/' + data_set + '/' + label) #读取列表，比如目标文件夹下的‘train’下label文件夹下的文件
            filenames.append([label, i])
            
    random.shuffle(filenames) #打乱数组中的顺序

##我的输入和真值在不同文件夹下同名，也可以不同名，但是打乱顺序以后需要进行相同的排序

def normalize_mi_ma(s,mi,ma,eps=1e-20,dtype=np.float32):
    x=(s-mi)/(ma-mi+eps)
    return x

def normalize(x,pmin=0.0,pmax=99.6,axis=None,eps=1e-20,dtype=np.float32):
    mi=np.percentile(x,pmin,axis=axis,keepdims=True)
    ma=np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x,mi,ma,eps=eps,dtype=dtype)


def get_data_tiff(data_dir,data_set, batch_size):
    global batch_index, filenames

    if len(filenames) == 0: get_filenames(data_dir,data_set)  #读取数据列表
    max = len(filenames)                          #得到file长度

    begin = batch_index                      #判断每一个batch的范围
    end = batch_index + batch_size

#    if end >= max:
#        end = max
#        q=0
#        batch_index = 0

    x_data = np.array([], np.float32)
    label_out=[]
    

    for i in range(begin, end):
        
        Input_Path = data_dir + data_set + '/input'+'/' + filenames[i][0]  #读取filenames中第i个数组的第一个元素
        label_out.append(filenames[i][0])
        
        input_tif=tiff.imread(Input_Path) #利用tifffile读取tiff文件，[depth,height,width],所以需要考虑是不是需要调整一下顺序

        inmin, inmax = input_tif.min(), input_tif.max()
        normal_input_tif=normalize(input_tif,2.0,99.9)#(input_tif-inmin)/(inmax-inmin)
        [d,h,w]=normal_input_tif.shape#last 100,933


        ##tensor_input=tf.convert_to_tensor(input_tif) ###将array转为tensor
        ####rshp_GT=tf.reshape(input_GT,[depth,height,width])  ##Tensorshape

        ##output=sess.run(tensor_input)  #通过sess得到网络的输出

        ##x_data = np.append(x_data, output)  #将输出保存到数组中
        x_data = np.append(x_data, normal_input_tif) #将输入保存到数组中

    if end>=max:
        end=max
        batch_index = 0
    else:
        batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, -1)

    return x_data_, label_out,[d,h,w]#返回数组中的值
