# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:31:08 2017

@author: linjian_sx
"""

from PIL import Image
import numpy as np
from math import log10
import os
import tensorflow as tf
import shutil
import random


def make_tf_label():
    width = 1280/2
    height = 720/2
    root_path = '/root/classify/'
    f = open('./valid_data/valid_list.txt','r')
    #tmp_list = f.readlines()
    #f.close()
    tmp_list = list()
    label_list = list()
    for item in f.readlines():
        info = item.strip('\n').split('\t')
        tmp_list.append(info[0])
        label_list.append(int(info[1]))
    writer = tf.python_io.TFRecordWriter(root_path+'valid_x2.tfrecords')
    for index,images in enumerate(tmp_list):
        print(images)
        raw_data = Image.open(images)
        raw_data = raw_data.resize((width,height))
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
