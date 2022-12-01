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
batch_index = 0
filenames = []

def get_filenames(data_dir,data_set): #获取文件名称
    global filenames
    filenames=os.listdir(os.path.join(data_dir, data_set,'input/'))
    print(filenames)
    filenames.sort()

##我的输入和真值在不同文件夹下同名，也可以不同名，但是打乱顺序以后需要进行相同的排序

def normalize_mi_ma(s,mi,ma,eps=1e-20,dtype=np.float32):
    if ma==0:
        x=s
    elif ma<2*mi:
        x=s/ma
    else:
        x=(s-mi)/(ma-mi+eps)
    return x

def normalize(x,pmin=0.0,pmax=99.6,axis=None,eps=1e-20,dtype=np.float32):
    mi=np.percentile(x,pmin,axis=axis,keepdims=True)
    ma=np.percentile(x,pmax,axis=axis,keepdims=True)
    return mi,ma,normalize_mi_ma(x,mi,ma,eps=eps,dtype=dtype)


def get_data_tiff(data_dir,data_set, batch_size,pmin,pmax):
    global batch_index, filenames

    if len(filenames) == 0: get_filenames(data_dir,data_set)  #读取数据列表
    max = len(filenames)                          #得到file长度

    begin = batch_index                      #判断每一个batch的范围
    end = batch_index + batch_size

    x_data = np.array([], np.float32)
    label_out=[]
    

    for i in range(begin, end):
        
        Input_Path = data_dir + data_set + '/input'+'/' + filenames[i]  #读取filenames中第i个数组的第一个元素
        label_out.append(filenames[i])
        
        input_tif=tiff.imread(Input_Path) #利用tifffile读取tiff文件，[depth,height,width],所以需要考虑是不是需要调整一下顺序

        inmin, inmax = input_tif.min(), input_tif.max()
        _,max_v,normal_input_tif=normalize(input_tif,pmin,pmax)#(input_tif-inmin)/(inmax-inmin)
        [d,h,w]=normal_input_tif.shape#last 100,933

        x_data = np.append(x_data, normal_input_tif) #将输入保存到数组中

    if end>=max:
        end=max
        batch_index = 0
    else:
        batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, -1)

    return np.squeeze(max_v),x_data_, label_out,[d,h,w]#返回数组中的值
