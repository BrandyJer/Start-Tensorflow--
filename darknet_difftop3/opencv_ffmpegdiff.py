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
import os, sys, cv2
import random
import time
from PIL import Image, ImageDraw, ImageFont

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    videoFilePath = r"/root/linjian/darknet_0/video/"
    temp_jpgPath = r"/root/temp_testdiff/"
    txt = open(r'/root/linjian/darknet_0/video/video116_ffmpegdiff.txt','a')
#    videoFile = cv2.VideoCapture(videoFilePath+'test2.mkv')

#    msec = int(videoFile.get(cv2.cv.CV_CAP_PROP_POS_MSEC))

#    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G') # motion-JPEG code
##6.14    out = videoFilePath+'video2_img_init'
##6.14    out_diff = videoFilePath+'video2_img_diff'
#    all_img = videoFilePath+'all_img-v2'
#    tiny_pick = videoFilePath+'tiny_pick-v3'
#    out = cv2.VideoWriter("/home/jz/JZ-working/e2ediff0_ZF_output.avi",
#                          fourcc, fps, size)
    i = 0
    num = 0
    jpg_list = os.listdir(temp_jpgPath)
    jpg_list.sort(key=lambda jpg:int(jpg.split('.')[0]))
    for jpg in jpg_list:
#        print"==========="
        find_obj = os.path.join(temp_jpgPath, jpg)
        image = cv2.imread(find_obj)
##ERROR        image = Image.open(find_obj)
##ERROR        image = np.array(image)
#        gray_img = cv2.CvtColor(image, cv2.COLOR_BGR2GRAY)
#        print ret
#        print image, type(image)
#        print(image.shape)
##6.14        cv2.imwrite(os.path.join(out, str(i)+'-'+time_str+'.jpg'), image)
        print("============== "+str(i)+" ============")
        if i == 0:
#            last_gray = gray_img
            last_BGR = image
        else:
#            frameDelta_gray = cv2.AbsDiff(last_gray, gray_img)
            frameDelta_BGR = cv2.absdiff(last_BGR, image)
            frameDelta_gray = cv2.cvtColor(frameDelta_BGR, cv2.COLOR_BGR2GRAY)

            mean_gray = np.mean(frameDelta_gray)
            mean_BGR = np.mean(frameDelta_BGR, axis=2)    #w*h
            BGR_nonZero = cv2.countNonZero(mean_BGR)
            gray_nonZero = cv2.countNonZero(frameDelta_gray)
#            print(frameDelta_gray.shape)
##6.14            cv2.imwrite(os.path.join(out_diff, str(i)+'-'+time_str+'.jpg'), frameDelta_BGR)
            print("-----"+str(mean_gray)+"-----")
#            print("-----"+str(mean_BGR)+"-----")
#            print("-----"+str(diff_nonZero)+"-----")
            txt.write(str(mean_gray)+' '+str(BGR_nonZero)+' '+str(gray_nonZero)+' '+jpg+'\n')
#            last_gray = gray_img
            last_BGR = image
            if i == 10000:
                break
        i = i+1
    txt.close()
#    videoFile.release()

#eval()


