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


def get_filenames(data_dir,data_set): 
    global filenames
    labels = []
    
    with open(data_dir +data_set + '/labels.txt') as f: 
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            labels += inner_list  
            
        for i, label in enumerate(labels):
            filenames.append([label, i])
            
    random.shuffle(filenames) 

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
    maxl = len(filenames)                          #得到file长度

    begin = batch_index                      #判断每一个batch的范围
    end = batch_index + batch_size

#    if end >= max:
#        end = max
#        q=0
#        batch_index = 0

    x_data1 = np.array([], np.float32)
    x_data2 = np.array([], np.float32)
    label_out=[]
    

    for i in range(begin, end):
        
        Input_Path1 = data_dir + data_set + '/input'+'/' + filenames[i][0]  #读取filenames中第i个数组的第一个元素
        Input_Path2 = data_dir + data_set + '/input2'+'/' + filenames[i][0]
        label_out.append(filenames[i][0])
        
        input_tif1=tiff.imread(Input_Path1) #利用tifffile读取tiff文件，[depth,height,width],所以需要考虑是不是需要调整一下顺序
        input_tif2=tiff.imread(Input_Path2)

        normal_input_tif1=normalize(input_tif1,0.0,99.9)#(input_tif-inmin)/(inmax-inmin)
        normal_input_tif2=normalize(input_tif2,0.0,99.9)#(input_tif-inmin)/(inmax-inmin)
        [d,h,w]=normal_input_tif1.shape#last 100,933

        x_data1 = np.append(x_data1, normal_input_tif1) #将输入保存到数组中
        x_data2 = np.append(x_data2, normal_input_tif2) #将输入保存到数组中

    if end>=maxl:
        end=maxl
        batch_index = 0
    else:
        batch_index += batch_size  # update index for the next batch
    x_data1_ = x_data1.reshape(batch_size, -1)
    x_data2_ = x_data2.reshape(batch_size, -1)

    return x_data1_,x_data2_, label_out,[d,h,w]#返回数组中的值
