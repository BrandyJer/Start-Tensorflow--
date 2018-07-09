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
import os, sys, cv2
import random
import time
from PIL import Image, ImageDraw, ImageFont
from net import tiny_darknet
import subprocess
import shutil
#from decode_tools import decode_from_tfrecords_eval
##use ffmpeg to decode 6min video to jpgs
##detect every img that diff_mean_gray > 5 

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def eval_video(w, h, im, obj_from, pick_out, num_6minjpg, time_jpg):
    with tf.Graph().as_default():
        with tf.variable_scope("model") as scope:
            thre = 0.91
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
            ckpt = tf.train.get_checkpoint_state(r"/root/linjian/darknet_0/models/try-linjian/JZ_data/new_0_wd4e5-0.15")
#####            ckpt = tf.train.get_checkpoint_state(r"/root/linjian/darknet_0/models/lj")
            saver.restore(sess, ckpt.all_model_checkpoint_paths[-2])
             
            l = sess.run([logits_eval],feed_dict={x: im})
#            print l, logits_eval
            p = l[0][0]
            if p[1]<= thre:
                HP = 1
                shutil.copy(obj_from, os.path.join(pick_out, str(num_6minjpg)+'-'+time_jpg+'.jpg'))
##6.13                cv2.imwrite(os.path.join(pick_out, str(num_th)+'-'+time_jpg+'.jpg'), im_pick)
##6.13                print('--------------The HP number is :'+str(num_th))
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
    temp_jpgPath = r"/root/temp_img/"
    jpg_OutPath = r"/root/temp_img/%5d.jpg"
    HP_OutPath = videoFilePath+"HP_116078561"
    start_time = 0
    end_time = 6*60
    shell_command_0 = r"ffmpeg -i "+videoFilePath
    shell_command_0 = shell_command_0+r"116078561.mkv -s 320x180 -sws_flags bilinear "
    videoFile = cv2.VideoCapture(videoFilePath+'116078561.mkv')
    totalFrameNumber = videoFile.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    rate = videoFile.get(cv2.cv.CV_CAP_PROP_FPS)
    video_time = int(totalFrameNumber/rate)
    videoFile.release()
    iter = 0   #How many 6min in a video
    i = 1 #prevent i yichu(line132), not use it ,just use it to identify the first frame of a video
    num = 0
    msec = 0
    mean_gray = 0.0
    while end_time <= video_time:
        shell_command = shell_command_0+r"-ss "+str(start_time)+r" -to "
        shell_command = shell_command+str(end_time)+' '+jpg_OutPath
        subprocess.call(shell_command, shell=True)   #creat 6min imgs
        print("=============== "+str(iter)+" =============== ")
        print("------ "+shell_command+" ------")

        img_list = os.listdir(temp_jpgPath)
        img_list.sort(key=lambda jpg:int(jpg.split('.')[0]))   #according the number to identify next frame
        for jpg in img_list:
            fps_jpg = int(jpg.split('.')[0])
#            msec = 40*fps_jpg+360000*iter   #1/25=0.04s=40ms
            msec = 40*fps_jpg+msec   #1/25=0.04s=40ms
            ms = msec%1000
            s = msec//1000
            min_ = s//60
            s_ = s%60
            time_min = str(min_).zfill(2)
            time_s = str(s_).zfill(2)
            time_ms = str(ms).zfill(3)
            time_str = time_min+'_'+time_s+'_'+time_ms

            find_obj = os.path.join(temp_jpgPath, jpg)
            image = cv2.imread(find_obj)

            if iter==0 and i==1:
                last_BGR = image
            else:
                frameDelta_BGR = cv2.absdiff(last_BGR, image)
                frameDelta_gray = cv2.cvtColor(frameDelta_BGR, cv2.COLOR_BGR2GRAY)
                mean_gray = np.mean(frameDelta_gray)
                last_BGR = image

            if mean_gray>5.0:
                image_tf = np.expand_dims(image, 0)
                print("-------------- "+jpg+" --------------")
                number, HP, YT =  eval_video(320, 180, image_tf, find_obj, HP_OutPath, fps_jpg, time_str)
                if number == 1:
                    num += 1
                    print('--------------How many times HP showed :'+str(num))
###            i += 1
        subprocess.call(r"rm -rf /root/temp_img/*.jpg", shell=True)
        start_time = end_time
        end_time +=360
        if end_time > video_time:
            end_time = video_time
        else:
            if end_time == video_time:
                end_time +=1
        iter += 1
#    videoFile.release()

#eval()


