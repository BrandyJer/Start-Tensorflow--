# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:54:02 2017

@author: linjian_sx
jiezhen change:
    line24 => 25
    line44 => 45
"""

from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import os, sys,cv2
import random
import time
from PIL import Image, ImageDraw, ImageFont
from net import tiny_darknet
#from decode_tools import decode_from_tfrecords_eval


#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval_video(w, h, im, im_pick, pick_out, num_th):
    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
            thre = 0.90
            HP = 0
            x = tf.placeholder(tf.float32, [1, h, w, 3])
#            x_img = tf.reshape(x, [1, int(720/4),int(1280/4),3])
#            labels = tf.placeholder(tf.int32, [None])
#            x_img = np.array([x_img])
            logits_eval = tiny_darknet(x,False)
            logits_eval = tf.reduce_mean(logits_eval,[1,2])
#            loss_eval =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_eval, logits=logits_eval)
            logits_eval = tf.nn.sigmoid(logits_eval)
             
            saver = tf.train.Saver(tf.all_variables())
            init = tf.initialize_all_variables()
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            sess = tf.Session()
            sess.run(init)
            
#            tf.train.start_queue_runners(sess=sess)
            ckpt = tf.train.get_checkpoint_state(r"/root/linjian/darknet_0/models/try-linjian/all_data/0_wd4e5-0.15")
#####            ckpt = tf.train.get_checkpoint_state(r"/root/linjian/darknet_0/models/lj")
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
             
            l = sess.run([logits_eval],feed_dict={x: im})
#            print l, logits_eval
            p = l[0][0]
            if p[0]>= thre:
                HP = 1
                cv2.imwrite(os.path.join(pick_out, str(num_th)+'.jpg'), im_pick)
                print('--------------The HP number is :'+str(num_th))
#                print('--------------How many times HP showed :'+str(num))
            else:
                HP = 0
            return HP, p[0], p[1]
#            cv2.waitKey(20)
#            if l[0][0]>= thre:
#                predict = 0     #HP
#            else:
#                predict = 1

if __name__ == '__main__':
    videoFilePath = r"/root/linjian/darknet_0/video/"
    txt = open(r'/root/linjian/darknet_0/video/dom-all_9999.txt','a')
    videoFile = cv2.VideoCapture(videoFilePath+'test.f4v')

    fps = videoFile.get(cv2.cv.CV_CAP_PROP_FPS)
    width_video = int(videoFile.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height_video = int(videoFile.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
#    msec = int(videoFile.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
#    size = (int(videoFile.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
#            int(videoFile.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

#    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G') # motion-JPEG code
    out = videoFilePath+'pick_img-9999all0.9'
    all_img = videoFilePath+'all_img-v2'
    tiny_pick = videoFilePath+'tiny_pick-v3'
#    out = cv2.VideoWriter("/home/jz/JZ-working/e2ediff0_ZF_output.avi",
#                          fourcc, fps, size)
    i = 0
    num = 0
#####    last_i = 0
#####    continue_yes = 0
#    print videoFile.isOpened()
    while videoFile.isOpened():
#        print"==========="
        ret, image = videoFile.read()
        msec = int(videoFile.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
        ms = msec%1000
        s = msec//1000
        min_ = s//60
        s_ = s%60
        time_min = str(min_).zfill(2)
        time_s = str(s_).zfill(2)
        time_ms = str(ms).zfill(3)
        time_str = time_min+':'+time_s+':'+time_ms
#        print ret
#        print image, type(image)
#        print(image.shape)
######        cv2.imwrite(os.path.join(all_img, str(i)+'.jpg'), image)
        image_cv = image
        image_tf = cv2.resize(image,(int(1280/4), int(720/4)), interpolation=cv2.INTER_AREA)
        image_tf = cv2.cvtColor(image_tf, cv2.COLOR_BGR2RGB)
        image_tf = np.expand_dims(image_tf, 0)
#        image = image.resize((int(1280/4), int(720/4)), Image.ANTIALIAS)
        print("============== "+str(i)+" ============")
        if image is None:
            break
        else:
            number, HP, YT =  eval_video(320, 180, image_tf, image_cv, out, i)
            txt.write(str(HP)+' '+str(YT)+' '+time_str+'\n')
            if number == 1:
                num += 1
                print('--------------How many times HP showed :'+str(num))
#####                if last_i == i-1:
#####                    continue_yes += 1
#####                else:
#####                    continue_yes = 0
#####                if continue_yes == 3:
#####                    image_5 = image
#####                if continue_yes == 6:
#####                    cv2.imwrite(os.path.join(tiny_pick, str(i-3)+'.jpg'), image_5)
#####                    contiune_yes = 0
#####                last_i = i            
        i = i+1
    txt.close()

#eval()

