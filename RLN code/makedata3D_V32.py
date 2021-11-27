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

FLAGS = tf.app.flags.FLAGS
#width =128#91#128 #define width
#height =128#239#128  #define height
#depth =128#386#128 #define depth
batch_index = 0
indexi=0
filenames = []

# user selection
#data_dir = '/home/liyue/newdata1/' #define data dictionary，include 'train','test',and include 'input' and 'ground truth' as subsictionary


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

def random_crop_64(x,y):
    w=max(0,(x.shape[0]-64)//7)
    h=max(0,(x.shape[1]-64)//7)
    d=max(0,(x.shape[2]-64)//7)
    x1=random.randint(0,w)*7#-64) #新的起始坐标
    y1=random.randint(0,h)*7#h-64)
    z1=random.randint(0,d)*7
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
    r_y=y[x1:x2,y1:y2,z1:z2]
    return r_x,r_y

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
#    print(mi)
    ma=np.percentile(x,pmax,axis=axis,keepdims=True)
#    print(ma)
    return normalize_mi_ma(x,mi,ma,eps=eps,dtype=dtype)

def data_aug_online(x,y,a,is_training):
    mean_whole=x.mean()#np.percentile(x,99.0,axis=None,keepdims=True)#x.max()
    if is_training:
#        a=random.randint(0,1)
        if a==0 or a==1:
            x1,y1=random_crop_64(x,y)
            while (x1.mean()<mean_whole*1):
                x1,y1=random_crop_64(x,y)
#                print(1)
        else:
            x1,y1=x,y
    return x1,y1

##我的输入和真值在不同文件夹下同名，也可以不同名，但是打乱顺序以后需要进行相同的排序

def get_data_tiff(data_dir, data_set, batch_size,is_training=False):
    global batch_index, filenames,maxl,indexi

    if len(filenames) == 0: get_filenames(data_dir, data_set)  #读取数据列表
    maxl = len(filenames)                          #得到file长度

    begin = 0                      #判断每一个batch的范围
    end =batch_size


    x_data = np.array([], np.float32)
    y_data = np.array([], np.float32) # zero-filled list for 'one hot encoding'
    label_out=[]
    
    a=random.randint(0,1)

        
    Input_Path = data_dir + data_set + '/input'+'/' + filenames[indexi][0]  #读取filenames中第i个数组的第一个元素
    GroundTruth_Path = data_dir + data_set + '/ground_truth'+'/' + filenames[indexi][0]
    label_out.append(filenames[indexi][0])
        
    input_tif1=tiff.imread(Input_Path) #利用tifffile读取tiff文件，[depth,height,width],所以需要考虑是不是需要调整一下顺序
    input_GT1=tiff.imread(GroundTruth_Path)

    for j in range(begin,end):
        input_tif,input_GT=data_aug_online(input_tif1,input_GT1,a,is_training)

        gtmin, gtmax,gtmean = input_GT.min(), input_GT.max(),input_GT.mean()


       # normal_input_tif,normal_gt_tif=data_aug_online(normal_input_tif,normal_gt_tif,a,is_training)
        normal_input_tif=normalize(input_tif,0.0,100)#last100
        #print(normal_input_tif.max())
#        print(normal_input_tif.max())
        normal_gt_tif=np.maximum(0,normalize(input_GT,1.0,99.9))
        [d,h,w]=normal_input_tif.shape
#        print(d,w,h)
        x_data = np.append(x_data, normal_input_tif) #将输入保存到数组中
        y_data = np.append(y_data, normal_gt_tif) #将真值保存在数组中
#        print(x_data.shape)

    if indexi+1>=maxl:
        indexi=0
    else:
        indexi=indexi+1  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, -1)
    y_data_ = y_data.reshape(batch_size, -1)

    return x_data_, y_data_ , label_out,[d,h,w]#返回数组中的值
