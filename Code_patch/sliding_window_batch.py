# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:10:33 2018

@author: jiezhen_sx
"""
import tensorflow as tf
#import os,os.path
#import shutil
#import cv2
#import numpy as np

def sliding_window(image, w, h):
    #slide a window across the image
    stride_h = h
    stride_w = w
    Img_H = image.shape[0]
    Img_W = image.shape[1]
    for h in range(0, Img_H, stride_h):
        for w in range(0, Img_W, stride_w):
            #yield the current window
            end_h = h+stride_h
            end_w = w+stride_w
            if end_h <= Img_H and end_w <= Img_W:
                yield image[h:end_h, w:end_w]
            else:
                star_h = h
                star_w = w
                if end_h > Img_H:
                    star_h = Img_H-stride_h
                if end_w > Img_W:
                    star_w = Img_W-stride_w
                yield image[star_h:star_h+stride_h, star_w:star_w+stride_w]
            
            
def output_batchpatch(raw_image):
    w = 256
    h = 180
    window_num = 0
    output_list = []
    for window in sliding_window(raw_image, w, h):
        window_tf = tf.cast(window, tf.float32)
        output_list.append(window_tf)
        window_num = len(output_list)
    batch_size = window_num
    output_list = tf.convert_to_tensor(output_list)
    filename_queue = tf.train.slice_input_producer([output_list], num_epochs=1, shuffle=False)
    patch_queue = tf.reshape(filename_queue[0], [h, w, 3])
    capacity = 10+3*batch_size
    image_batch= tf.train.batch([patch_queue],
                                batch_size=batch_size, 
                                num_threads=3, 
                                capacity=capacity)
    return image_batch, window_num

##cut windsize patch form anysize image
#test_allwindow = []
#img_num = 0
#img_path = r'D:\jiezhen\workshop\Huaping_jz\JZ_patch\test'
##save_patch = r'D:\jiezhen\workshop\Huaping_jz\JZ_patch\test_sliding'
#
#for img_name in os.listdir(img_path):
#    path = os.path.join(img_path, img_name)
#    img = cv2.imread(path)
#    sess = tf.Session()
#    init = tf.initialize_all_variables()
#    coord = tf.train.Coordinator()
#    with sess.as_default():
#        sess.run(init)
#        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#        patch_list, all_windows = output_batchpatch(img)
#        print(patch_list)
#    coord.request_stop()
#    coord.join(threads)
##    test_allwindow.append(all_windows)
##    for i in range(all_windows):
##        cv2.imwrite(os.path.join(save_patch, str(img_num)+'_'+str(i)+'.jpg'),
##                    patch_list[i])
##    img_num += 1
