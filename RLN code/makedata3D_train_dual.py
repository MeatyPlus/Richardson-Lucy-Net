#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:57:35 2020

@author: lee
"""

# A script to load images and make batch.

import os
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import random
import tifffile as tiff

FLAGS = tf.app.flags.FLAGS
batch_index = 0
indexi=0
filenames = []

# user selection
#data_dir = '/home/liyue/newdata1/'

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

def random_crop_64(x,x_1,y):
    w=max(0,(x.shape[0]-64)//8)
    h=max(0,(x.shape[1]-64)//8)
    d=max(0,(x.shape[2]-64)//8)
    x1=random.randint(0,max(0,w-1))*8#-64) #新的起始坐标
    y1=random.randint(0,max(0,h-1))*8#h-64)
    z1=random.randint(0,max(0,d-1))*8
    x2=x1+64 #新的结尾坐标
    y2=y1+64
    z2=z1+64
    if x2>x.shape[0]:
        x1=0
        x2=x.shape[0]-1
    if y2>x.shape[1]:
        y1=0
        y2=x.shape[1]-1
    if z2>x.shape[2]:
        z1=0
        z2=x.shape[2]-1
    r_x=x[x1:x2,y1:y2,z1:z2]
    r_x1=x_1[x1:x2,y1:y2,z1:z2]
    r_y=y[x1:x2,y1:y2,z1:z2]
    return r_x,r_x1,r_y

def random_crop_32(x,y):
    w=x.shape[0]
    h=x.shape[1]
    d=x.shape[2]
    x1=random.randint(0,w-32) #新的起始坐标
    y1=random.randint(0,h-32)
    z1=random.randint(0,d-32)
    x2=x1+32 #新的结尾坐标
    y2=y1+32
    z2=z1+32
    r_x=x[x1:x2,y1:y2,z1:z2]
    r_y=y[x1:x2,y1:y2,z1:z2]
    return r_x,r_y

def random_crop_16(x,y):
    w=x.shape[0]
    h=x.shape[1]
    d=x.shape[2]
    x1=random.randint(0,w-16) #新的起始坐标
    y1=random.randint(0,h-16)
    z1=random.randint(0,d-16)
    x2=x1+16 #新的结尾坐标
    y2=y1+16
    z2=z1+16
    r_x=x[x1:x2,y1:y2,z1:z2]
    r_y=y[x1:x2,y1:y2,z1:z2]
    return r_x,r_y
    
def normalize_mi_ma(s,mi,ma,eps=1e-20,dtype=np.float32):
    x=(s-mi)/(ma-mi+eps)
    return x

def normalize(x,pmin=0.0,pmax=99.6,axis=None,eps=1e-20,dtype=np.float32):
    mi=np.percentile(x,pmin,axis=axis,keepdims=True)
    ma=np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x,mi,ma,eps=eps,dtype=dtype)

def data_aug_online(x,x_1,y,a,is_training):
    mean_whole=x.mean()
    if is_training:
        if a==0 or a==1:
            x1,x1_1,y1=random_crop_64(x,x_1,y)
            while (x1.mean()<mean_whole*0.8):
                x1,x1_1,y1=random_crop_64(x,x_1,y)
               # print(1)
        else:
            x1,x1_1,y1=x,y
    return x1,x1_1,y1


def get_data_tiff(data_dir, data_set, batch_size,is_training=False):
    global batch_index, filenames,maxl,indexi

    if len(filenames) == 0: get_filenames(data_dir, data_set)  #读取数据列表
    maxl = len(filenames)                          #得到file长度

    begin = batch_index                      #判断每一个batch的范围
    end = batch_index + batch_size


    x_data1 = np.array([], np.float32)
    x_data2 = np.array([], np.float32)
    y_data = np.array([], np.float32) # zero-filled list for 'one hot encoding'
    label_out=[]

    Input_Path1 = data_dir + data_set + '/input'+'/' + filenames[indexi][0]  #读取filenames中第i个数组的第一个元素
    Input_Path2 = data_dir + data_set + '/input2'+'/' + filenames[indexi][0]
    GroundTruth_Path = data_dir + data_set + '/ground_truth'+'/' + filenames[indexi][0]
    label_out.append(filenames[indexi][0])
        
    input_tif11=tiff.imread(Input_Path1) #利用tifffile读取tiff文件，[depth,height,width],所以需要考虑是不是需要调整一下顺序
    input_tif21=tiff.imread(Input_Path2)
    input_GT1=tiff.imread(GroundTruth_Path)
    
    a=random.randint(0,1)
    for i in range(begin, end):
        input_tif1,input_tif2,input_GT=data_aug_online(input_tif11,input_tif21,input_GT1,a,is_training)

        gtmin, gtmax,gtmean = input_GT.min(), input_GT.max(),input_GT.mean()



        normal_input_tif1=normalize(input_tif1,0.0,99.9)#last100
        normal_input_tif2=normalize(input_tif2,0.0,99.9)
        normal_gt_tif=np.maximum(0,normalize(input_GT,1.0,99.9))
        [d,h,w]=normal_input_tif1.shape
#        print(d,w,h)
        x_data1 = np.append(x_data1, normal_input_tif1) #将输入保存到数组中
        x_data2 = np.append(x_data2, normal_input_tif2) #将输入保存到数组中
        y_data = np.append(y_data, normal_gt_tif) #将真值保存在数组中
#        print(x_data.shape)

    if indexi+1>=maxl:
        indexi=0
    else:
        indexi=indexi+1   # update index for the next batch
    x_data1_ = x_data1.reshape(batch_size, -1)
    x_data2_ = x_data2.reshape(batch_size, -1)
    y_data_ = y_data.reshape(batch_size, -1)

    return x_data1_, x_data2_,y_data_ , label_out,[d,h,w]#返回数组中的值



def get_data_tiff_VL(data_dir, data_set, batch_size,is_training=False):
    global batch_index, filenames,maxl,indexi

    if len(filenames) == 0: get_filenames(data_dir, data_set)  #读取数据列表
    maxl = len(filenames)                          #得到file长度

    begin = batch_index                      #判断每一个batch的范围
    end = batch_index + batch_size


    x_data1 = np.array([], np.float32)
    x_data2 = np.array([], np.float32)
    y_data = np.array([], np.float32) # zero-filled list for 'one hot encoding'
    label_out=[]

    Input_Path1 = data_dir + data_set + '/input'+'/' + filenames[indexi][0]  #读取filenames中第i个数组的第一个元素
    Input_Path2 = data_dir + data_set + '/input2'+'/' + filenames[indexi][0]
    GroundTruth_Path = data_dir + data_set + '/ground_truth'+'/' + filenames[indexi][0]
    label_out.append(filenames[indexi][0])
        
    input_tif1=tiff.imread(Input_Path1) #利用tifffile读取tiff文件，[depth,height,width],所以需要考虑是不是需要调整一下顺序
    input_tif2=tiff.imread(Input_Path2)
    input_GT=tiff.imread(GroundTruth_Path)
    
    a=random.randint(0,1)
    for i in range(begin, end):
        gtmin, gtmax,gtmean = input_GT.min(), input_GT.max(),input_GT.mean()



        normal_input_tif1=normalize(input_tif1,0.0,99.9)#last100
        normal_input_tif2=normalize(input_tif2,0.0,99.9)
        normal_gt_tif=np.maximum(0,normalize(input_GT,1.0,99.9))
        [d,h,w]=normal_input_tif1.shape

        #print(d,w,h)
        x_data1 = np.append(x_data1, normal_input_tif1) #将输入保存到数组中
        x_data2 = np.append(x_data2, normal_input_tif2) #将输入保存到数组中
        y_data = np.append(y_data, normal_gt_tif) #将真值保存在数组中
#        print(x_data.shape)

    if indexi+1>=maxl:
        indexi=0
    else:
        indexi=indexi+1   # update index for the next batch
    x_data1_ = x_data1.reshape(batch_size, -1)
    x_data2_ = x_data2.reshape(batch_size, -1)
    y_data_ = y_data.reshape(batch_size, -1)

    return x_data1_, x_data2_,y_data_ , label_out,[d,h,w]#返回数组中的值
