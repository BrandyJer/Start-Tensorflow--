# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:31:08 2017

@author: jiezhen_sx
"""

from PIL import Image
import numpy as np
from math import log10
import os, os.path
import tensorflow as tf
import shutil
import random


def make_tf_label():
    width = 256
    height = 180
    root_path = r'/root/linjian/darknet_patch/TF_data/'
    f = open(r'/root/linjian/darknet_patch/raw_data/train/train_patch0.txt','r')
    #tmp_list = f.readlines()
    #f.close()
    tmp_list = list()
    label_list = list()
    for item in f.readlines():
        info = item.strip('\n').split('\t')
        tmp_list.append(info[0])
        label_list.append(int(info[1]))
    writer = tf.python_io.TFRecordWriter(root_path+'train_patch0.tfrecords')
    for index,images in enumerate(tmp_list):
#        images_path = images.split('\\')
#        dir_here = os.getcwd()
#        dir_2 = os.path.join(dir_here, images_path[-2]) #label
#        dir_1 = os.path.join(dir_2, images_path[-1]) #image name
#        print(dir_1)
        print(images)
        raw_data = Image.open(images)
#        raw_data = raw_data.resize((width,height))
        data = raw_data.tobytes()
        label = label_list[index]
        example = tf.train.Example(features = tf.train.Features(feature={
                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                                "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
                                "img_name":tf.train.Feature(bytes_list=tf.train.BytesList(value=[images]))      
                }))
        writer.write(example.SerializeToString())
    writer.close()
    return 1


 
make_tf_label() 
